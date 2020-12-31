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
package edu.cmu.lti.oaqa.flexneuart.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import no.uib.cipr.matrix.DenseVector;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.external.TextEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.letor.external.TextEntryRaw;
import edu.cmu.lti.oaqa.flexneuart.letor.external.WordEntryInfo;
import edu.cmu.lti.oaqa.flexneuart.letor.external.ExternalScorer.Client;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

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

  public static final String NORM_BY_QUERY_LEN = "normByQueryLen";
  
  public static String EXTR_TYPE = "externalThrift";
  
  public static String FEAT_QTY = "featureQty";

  public static String HOST = "host";
  public static String PORT = "port";
  public static String PORT_LIST = "portList";
  
  public static String POSITIONAL = "useWordSeq";
  
  public static String UNK_WORD = "unkWord";
  
  public static String PARSED_AS_RAW = "sendParsedAsRaw";

  public FeatExtractorExternalApacheThrift(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    // getReqParamStr throws an exception if the parameter is not defined
    
    mFeatQty = conf.getReqParamInt(FEAT_QTY);

    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());

    int port = conf.getParam(PORT, -1);
    String portArrStr = conf.getParam(PORT_LIST, null);
    
    if (port >= 0) {
      if (portArrStr != null) {
        throw new Exception("One shouldn't specify both " + PORT + " and " + PORT_LIST);
      }
      mPortArr = new int[1];
      mPortArr[0] = port;
    } else {
      String tmp[] = StringUtils.splitNoEmpty(portArrStr, ",");
      mPortArr = new int[tmp.length];
      for (int k = 0; k < tmp.length; ++k) {
        try {
          mPortArr[k] = Integer.parseInt(tmp[k]);
        } catch (NumberFormatException e) {
          throw new Exception("Invalid port number in: " + portArrStr);
        }
      }
    }
    
    
    mHost = conf.getReqParamStr(HOST);
    
    mUnkWord = conf.getReqParamStr(UNK_WORD);
    
    mNormByQueryLen = conf.getParamBool(NORM_BY_QUERY_LEN);
    
    mTextAsRaw = conf.getParamBool(PARSED_AS_RAW);
    
    mUseWordSeq = conf.getParamBool(POSITIONAL);
    
    mSimilObj = new  BM25SimilarityLuceneNorm(BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                              BM25SimilarityLucene.DEFAULT_BM25_B, 
                                              mFieldIndex);
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
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = new HashMap<String, DenseVector>();
    
    /*
     * This is not super-efficient. However, implementing release/acquire without closing socket
     * and caching sockets somehow leads to a dead-lock sometimes. In any case, openining/closing
     * connection is quite fast compared to re-ranking the output of a single query using
     * SOTA neural models. 
     */
    int port = getPort();
    TTransport transp = new TSocket(mHost, port);
    transp.open();
     
    try {
      Client clnt = new Client(new TBinaryProtocol(transp));
      
      res = initResultSet(cands, getFeatureQty()); 

      Map<String, List<Double>> scores = null;
      
      String queryId = queryData.get(Const.TAG_DOCNO);
      
      if (queryId == null) {
        throw new Exception("Undefined query ID!");
      }
      
      if (mFieldIndex.isRaw()) {
        logger.info("Sending raw/unparsed entry, port: " + port);
        String queryTextRaw = queryData.get(getQueryFieldName());
        if (queryTextRaw == null) return res;
        
        ArrayList<TextEntryRaw> docEntries = new ArrayList<>();
        
        for (CandidateEntry e : cands) {
          String docEntryRaw = mFieldIndex.getDocEntryRaw(e.mDocId);

          if (docEntryRaw == null) {
            throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
          }

          docEntries.add(new TextEntryRaw(e.mDocId, docEntryRaw));
        }
        scores = clnt.getScoresFromRaw(new TextEntryRaw(queryId, queryTextRaw), docEntries);
      } else {        
        if (mTextAsRaw) {
          // This can be a lot faster on the Python side!
          logger.info("Sending parsed entry in the unparsed/raw format, port: " + port);
          String queryTextRaw = queryData.get(getQueryFieldName());
          if (queryTextRaw == null) return res;
          
          ArrayList<TextEntryRaw> docEntries = new ArrayList<>();
          
          for (CandidateEntry e : cands) {
            String docEntryRaw = mFieldIndex.getDocEntryParsedText(e.mDocId);

            if (docEntryRaw == null) {
              throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
            }

            docEntries.add(new TextEntryRaw(e.mDocId, docEntryRaw));
          }
          scores = clnt.getScoresFromRaw(new TextEntryRaw(queryId, queryTextRaw), docEntries);
          
        } else {
          logger.info("Sending parsed entry, port: " + port);
          DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryData);
          if (queryEntry == null)
            return res;
          
          TextEntryParsed queryTextEntry = createTextEntryParsed(queryId, queryEntry);
          ArrayList<TextEntryParsed> docTextEntries = new ArrayList<>();
          for (CandidateEntry e : cands) {
            DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(e.mDocId);

            if (docEntry == null) {
              throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
            }

            docTextEntries.add(createTextEntryParsed(e.mDocId, docEntry));
          }
          scores = clnt.getScoresFromParsed(queryTextEntry, docTextEntries);
        }
      }
      
      String query = queryData.get(Const.TEXT_FIELD_NAME);
      if (query == null) {
        query = "";
      }
      float queryQty = StringUtils.splitOnWhiteSpace(query).length;
      if (queryQty <= 1) {
        queryQty = 1;
      }
      float invQueryQty = 1.0f / queryQty;
      if (mNormByQueryLen) {
        logger.info(String.format("Query: '%s' Normalization factor: %f", query, invQueryQty));
      }
      
      for (CandidateEntry e : cands) {
        List<Double> scoreList = scores.get(e.mDocId);

        if (scoreList == null) {
          throw new Exception("Inconsistent data or bug: can't find a score for doc id ='" + e.mDocId + "'");
        }
        if (scoreList.size() != mFeatQty) {
          throw new Exception("Inconsistent data or bug for doc id ='" + e.mDocId + "' expected " + mFeatQty + 
              " features, but got: " + scoreList.size());
        }
        
        DenseVector scoreVect = new DenseVector(mFeatQty);
        
        int idx = 0;
        
        for (double v : scoreList) {
          if (mNormByQueryLen) {
            v *= invQueryQty;
          }
          scoreVect.set(idx, v);
          idx++;
        }
        
        res.put(e.mDocId, scoreVect);
      }
    } catch (Exception e) {
      logger.error("Caught an exception:" + e);
      transp.close();
      throw e;
    }    
    
    transp.close();
    
    return res;
  }
  
  final TFIDFSimilarity              mSimilObj;
  final ForwardIndex                 mFieldIndex;
  final String                       mHost;
  final int                          mPortArr[];
  final String                       mUnkWord;
  
  final int                          mFeatQty;
  final boolean                      mUseWordSeq;
  
  final boolean                      mNormByQueryLen;
  
  final boolean                      mTextAsRaw;
  
  int                                mPortIdToUse = -1; // getPort will first increment it
  
  synchronized int getPort() {
    mPortIdToUse = (mPortIdToUse + 1) % mPortArr.length;
    return mPortArr[mPortIdToUse];
  }
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public int getFeatureQty() {
    return mFeatQty;
  }
}

  