/*
 *  Copyright 2014+ Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.knn4qa.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import no.uib.cipr.matrix.DenseVector;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.letor.external.TextEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.letor.external.TextEntryRaw;
import edu.cmu.lti.oaqa.knn4qa.letor.external.WordEntryInfo;
import edu.cmu.lti.oaqa.knn4qa.letor.external.ExternalScorer.Client;

/**
 * A single-field feature extractor that calls an external code (via Apache Thrift) to compute 
 * one or more scores. The number of scores per entry is fixed and is specified as a parameter
 * for this extractor.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtractorExternalApacheThrift extends SingleFieldFeatExtractor {
  private static final Logger logger = LoggerFactory.getLogger(FeatExtractorExternalApacheThrift.class);
  
  public static String EXTR_TYPE = "externalThrift";
  
  public static String FEAT_QTY = "featureQty";

  public static String HOST = "host";
  public static String PORT = "port";
  
  public static String POSITIONAL = "useWordSeq";
  
  public static String UNK_WORD = "unkWord";

  public FeatExtractorExternalApacheThrift(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    // getReqParamStr throws an exception if the parameter is not defined
    
    mFeatQty = conf.getReqParamInt(FEAT_QTY);

    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());

    mPort = conf.getReqParamInt(PORT);
    mHost = conf.getReqParamStr(HOST);
    
    mUnkWord = conf.getReqParamStr(UNK_WORD);
    
    mUseWordSeq = conf.getParamBool(POSITIONAL);
    
    mSimilObj = new  BM25SimilarityLuceneNorm(BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                              BM25SimilarityLucene.DEFAULT_BM25_B, 
                                              mFieldIndex);
  }
  
  /**
   * getSocket() retrieves the first unused socket, or creates a new one.
   * Note that a single client and socket cannot be re-used across threads, b/c
   * each client/thread needs to use a separate socket.
   */
  private synchronized TTransport acquireTransport() throws TTransportException {
    if (!mFreeTransports.isEmpty()) {
      int sz = mFreeTransports.size();
      TTransport ret = mFreeTransports.get(sz-1);
      mFreeTransports.remove(sz-1);
      return ret;
    }
    
    TTransport serviceTransp = new TSocket(mHost, mPort);
    serviceTransp.open();
    
    return serviceTransp;
  }
  
  private synchronized void releaseTransport(TTransport c) {
    mFreeTransports.add(c);
  }

  /**
   * Create a parsed-text entry. Unknown words are going to be replaced with a special token.
   * 
   * @param entryId   entry ID, e.g., document ID.
   * @param docEntry  document entry data point
   * @return
   * @throws Exception 
   */
  TextEntryParsed createTextEntryParsed(String entryId, DocEntryParsed docEntry) throws Exception {
    ArrayList<WordEntryInfo> wentries = new ArrayList<WordEntryInfo>();
    
    if (mUseWordSeq) {
      if (null == docEntry.mWordIdSeq) {
        throw new Exception("Configuration error: positional info is not stored for field: '" + getIndexFieldName() + "'");
      }
      for (int wid : docEntry.mWordIdSeq) {
        if (wid >= 0) {
          float idf = mSimilObj.getIDF(mFieldIndex, wid);
          wentries.add(new WordEntryInfo(mFieldIndex.getWord(wid), idf, 1));
        } else {
          wentries.add(new WordEntryInfo(mUnkWord, 0, 1));
        }
      }
    } else {
      for (int k = 0; k < docEntry.mWordIds.length; ++k) {
        int wid = docEntry.mWordIds[k];
        int qty = docEntry.mQtys[k];
        if (wid >= 0) {
          float idf = mSimilObj.getIDF(mFieldIndex, wid);
          wentries.add(new WordEntryInfo(mFieldIndex.getWord(wid), idf, qty));
        } else {
          wentries.add(new WordEntryInfo(mUnkWord, 0, qty));
        }
       }
    }
    return new TextEntryParsed(entryId, wentries); 
  }
  
  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = new HashMap<String, DenseVector>();
    
    TTransport transp = acquireTransport();
     
    try {
      Client clnt = new Client(new TBinaryProtocol(transp));
      
      res = initResultSet(arrDocIds, getFeatureQty()); 

      Map<String, List<Double>> scores = null;
      
      String queryId = queryData.get(Const.TAG_DOCNO);
      
      if (queryId == null) {
        throw new Exception("Undefined query ID!");
      }
      
      if (mFieldIndex.isRaw()) {
        String queryTextRaw = queryData.get(getQueryFieldName());
        if (queryTextRaw == null) return res;
        
        ArrayList<TextEntryRaw> docEntries = new ArrayList<>();
        
        for (String docId : arrDocIds) {
          String docEntryRaw = mFieldIndex.getDocEntryRaw(docId);

          if (docEntryRaw == null) {
            throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
          }

          docEntries.add(new TextEntryRaw(docId, docEntryRaw));
        }
        scores = clnt.getScoresFromRaw(new TextEntryRaw(queryId, queryTextRaw), docEntries);
      } else {
        DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryData);
        if (queryEntry == null) return res;
        
        TextEntryParsed queryTextEntry = createTextEntryParsed(queryId, queryEntry);
        ArrayList<TextEntryParsed> docTextEntries = new ArrayList<>();
        for (String docId : arrDocIds) {
          DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(docId);

          if (docEntry == null) {
            throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
          }

          docTextEntries.add(createTextEntryParsed(docId, docEntry));
        }
        scores = clnt.getScoresFromParsed(queryTextEntry, docTextEntries);
      }
      
      
      for (String docId : arrDocIds) {
        List<Double> scoreList = scores.get(docId);

        if (scoreList == null) {
          throw new Exception("Inconsistent data or bug: can't find a score for doc id ='" + docId + "'");
        }
        if (scoreList.size() != mFeatQty) {
          throw new Exception("Inconsistent data or bug for doc id ='" + docId + "' expected " + mFeatQty + 
              " features, but got: " + scoreList.size());
        }
        
        DenseVector scoreVect = new DenseVector(mFeatQty);
        
        int idx = 0;
        
        for (double v : scoreList) {
          scoreVect.set(idx, v);
          idx++;
        }
        
        res.put(docId, scoreVect);
      }
    } catch (Exception e) {
      logger.error("Caught an exception:" + e);
      releaseTransport(transp);
      throw e;
    }    
    
    releaseTransport(transp);
    
    return res;
  }
  
  final TFIDFSimilarity              mSimilObj;
  final ForwardIndex                 mFieldIndex;
  final String                       mHost;
  final int                          mPort;
  final String                       mUnkWord;
  
  final int                          mFeatQty;
  final boolean                      mUseWordSeq;

  private ArrayList<TTransport>      mFreeTransports = new ArrayList<>();
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public int getFeatureQty() {
    return mFeatQty;
  }
}

  