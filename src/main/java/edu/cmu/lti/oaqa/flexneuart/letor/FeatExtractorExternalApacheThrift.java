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
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;

class HostPort {
  public final String mHost;
  public final int    mPort;
  
  protected HostPort(String addr) throws Exception {
    // This parsing is a bit simplistic, but it should work as long as the sticks to the 
    // following notation: <address or host name without http and/or https prefix> : <numeric port>
    int sepChar = addr.indexOf(':');
    if (sepChar < 0) {
      throw new Exception("No port separator : is present in the host address: '" + addr + "'");
    }
    mHost = addr.substring(0, sepChar);
    String port = addr.substring(sepChar + 1);
    try {
      mPort = Integer.valueOf(port);
    } catch (NumberFormatException e) {
      throw new Exception("Port is not ineger in the host address: '" + addr + "'");
    }
  }
  
  @Override
  public String toString() {
    return "host: " + mHost + " port: " + mPort;
    
  }
}

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

  public static String EXTR_TYPE = "ExternalThrift";
  
  public static String FEAT_QTY = "featureQty";

  public static String HOST_LIST = "hostList";
  
  public static String POSITIONAL = "useWordSeq";
  
  public static String UNK_WORD = "unkWord";
  
  public static String PARSED_AS_RAW = "sendParsedAsRaw";

  public FeatExtractorExternalApacheThrift(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    super(resMngr, conf);
    // getReqParamStr throws an exception if the parameter is not defined
    
    mFeatQty = conf.getReqParamInt(FEAT_QTY);

    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());

    for (String hostStr : conf.getParamNestedConfig(HOST_LIST).getParamStringArray()) {
      mHostArr.add(new HostPort(hostStr));
    }
    
    mUnkWord = conf.getReqParamStr(UNK_WORD);
    
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
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields)
      throws Exception {
    HashMap<String, DenseVector> res = new HashMap<String, DenseVector>();
    
    /*
     * This is not super-efficient. However, implementing release/acquire without closing socket
     * and caching sockets somehow leads to a dead-lock sometimes. In any case, openining/closing
     * connection is quite fast compared to re-ranking the output of a single query using
     * SOTA neural models. 
     */
    HostPort hostInfo = getHostInfo();
    TTransport transp = new TSocket(hostInfo.mHost, hostInfo.mPort);
    transp.open();
     
    try {
      Client clnt = new Client(new TBinaryProtocol(transp));
      
      res = initResultSet(cands, getFeatureQty()); 

      Map<String, List<Double>> scores = null;
      
      String queryId = queryFields.mEntryId;  
      if (queryId == null) {
        throw new Exception("Undefined query ID!");
      }
      
      if (mFieldIndex.isTextRaw()) {
        logger.info("Sending raw/unparsed entry: " + hostInfo);
        String queryTextRaw = queryFields.getString(getQueryFieldName());
        if (queryTextRaw == null) {
          warnEmptyQueryField(logger, EXTR_TYPE, queryId);
          return res;
        }
        
        ArrayList<TextEntryRaw> docEntries = new ArrayList<>();
        
        for (CandidateEntry e : cands) {
          String docEntryRaw = mFieldIndex.getDocEntryTextRaw(e.mDocId);

          if (docEntryRaw == null) {
            throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
          }

          docEntries.add(new TextEntryRaw(e.mDocId, docEntryRaw));
        }
        scores = clnt.getScoresFromRaw(new TextEntryRaw(queryId, queryTextRaw), docEntries);
      } else if (mFieldIndex.isParsed()) {        
        if (mTextAsRaw) {
          // This can be a lot faster on the Python side!
          logger.info("Sending parsed entry in the unparsed/raw format to " + hostInfo);
          String queryTextRaw = queryFields.getString(getQueryFieldName());
          if (queryTextRaw == null) {
            warnEmptyQueryField(logger, EXTR_TYPE, queryId);
            return res;
          }
          
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
          logger.info("Sending parsed entry to " + hostInfo);
          DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryFields);
          if (queryEntry == null) {
            warnEmptyQueryField(logger, EXTR_TYPE, queryId);
            return res;
          }
          
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
      } else {
        throw new RuntimeException(EXTR_TYPE + " works only with parsed or raw text fields, " + 
                                   "but " + this.getIndexFieldName() + " is binary");
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
  final ArrayList<HostPort>          mHostArr = new ArrayList<HostPort>();
  final String                       mUnkWord;
  
  final int                          mFeatQty;
  final boolean                      mUseWordSeq;
  
  final boolean                      mTextAsRaw;
  
  int                                mHostIdToUse = -1; // getPort will first increment it
  
  synchronized HostPort getHostInfo() {
    mHostIdToUse = (mHostIdToUse + 1) % mHostArr.size();
    return mHostArr.get(mHostIdToUse);
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

  