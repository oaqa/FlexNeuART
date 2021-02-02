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
package edu.cmu.lti.oaqa.flexneuart.cand_providers;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.EvalUtils;

/**
 * A candidate provider that reads from a previously generated TREC-style run.
 * The run should be placed into the same directory with a query file. Furthermore,
 * 1. the input query is supposed match exactly one of the provided queries.
 *    when there's no match, a warning is produced.
 * 2. all queries are IDs must be unique
 * 3. the query is matched to the respective run using a query ID.
 * 
 * @author Leonid Boytsov
 *
 */
public class TrecRunCandidateProvider extends CandidateProvider {
  final static Logger logger = LoggerFactory.getLogger(TrecRunCandidateProvider.class);

  public final static String QUERY_FILE_PARAM = "queryFile";
  public final static String TREC_RUN_FILE_PARAM = "trecRunFile";
  
  /***
   * Constructor.
   * 
   * @param trecRunDirName  the name of the directory with a question file
   * @param addConf
   * @throws Exception
   */
  public TrecRunCandidateProvider(String trecRunDirName, CandProvAddConfig addConf) throws Exception {
    String queryFileName = addConf.getReqParamStr(QUERY_FILE_PARAM);
    String trecRunFileName = addConf.getReqParamStr(TREC_RUN_FILE_PARAM);
    
    String fullQueryFile = new File(trecRunDirName, queryFileName).getAbsolutePath();
    
    int queryQty = 0;
    try (DataEntryReader inp = new DataEntryReader(fullQueryFile)) {
      Map<String, String> queryFields = null;      
      
      while ((queryFields = inp.readNext()) != null) {
         
        String text = queryFields.get(Const.TEXT_FIELD_NAME);
        String queryId = queryFields.get(Const.TAG_DOCNO);
        
        if (queryId == null) {
          logger.info("Null queryID, query #: " + (queryQty+1));
          continue;
        }
        
        if (text == null) {
          logger.info("Null text query ID: " + queryId);
          continue;
        }
        
        mQueryTextToId.put(text.trim(), queryId);
        
        ++queryQty;
        if (queryQty % 1000 == 0) logger.info("Read " + queryQty + " documents from " + fullQueryFile);
      }
    }
    
    logger.info("Read " + queryQty + " documents from " + fullQueryFile);
    
    String fullTrecRunFile = new File(trecRunDirName, trecRunFileName).getAbsolutePath();
    
    mTrecRuns = EvalUtils.readTrecResults(fullTrecRunFile);
    
    logger.info("Read " + mQueryTextToId.size() + " queries from " + fullTrecRunFile);
  }

  @Override
  public boolean isThreadSafe() {
    // True, b/c reading the data is static and read-only access is safe
    return true;
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }  
  
  @Override
  public CandidateInfo getCandidates(int queryNum, Map<String, String> queryData, int maxQty) throws Exception {
    String queryID = queryData.get(ID_FIELD_NAME);
    if (null == queryID) {
      throw new Exception(
          String.format("Query id (%s) is undefined for query # %d",
                        ID_FIELD_NAME, queryNum));
    }        
    
    String text = queryData.get(QUERY_FIELD_NAME);
    if (null == text) {
      throw new Exception(
          String.format("Query (%s) is undefined for query # %d",
                        QUERY_FIELD_NAME, queryNum));
    }
    text = text.trim();
    String cacheQueryID = mQueryTextToId.get(text);
    if (cacheQueryID == null) {
      logger.warn("Ignoring  query " + queryID + " because matching by text failed");
      return mEmpty;
    }
    ArrayList<CandidateEntry> resArr = mTrecRuns.get(cacheQueryID);
    if (resArr == null) {
      logger.warn("Ignoring  query " + queryID + 
                 " because trec run contains not entry for cached query ID " + cacheQueryID);
      return mEmpty;
    }
    
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
    Arrays.sort(results);
        
    return new CandidateInfo(results.length, results);
  }
  
  private CandidateInfo mEmpty = new CandidateInfo(0, new CandidateEntry[0]);
  private HashMap<String, String>   mQueryTextToId = new HashMap<String, String>();
  private HashMap<String, ArrayList<CandidateEntry>>  mTrecRuns = null;
}
