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
 * The results are matched using a query ID, so it must be exactly the same as in 
 * the query that originally generated that TREC run.
 * 
 * @author Leonid Boytsov
 *
 */
public class TrecRunCandidateProvider extends CandidateProvider {
  final static Logger logger = LoggerFactory.getLogger(TrecRunCandidateProvider.class);
  
  /***
   * Constructor.
   * 
   * @param trecFileName  the name of the trec run file (can be gz or bz2 compressed)
   * 
   * @throws Exception
   */
  public TrecRunCandidateProvider(String trecFileName) throws Exception {

    mTrecRuns = EvalUtils.readTrecResults(trecFileName);
    
    logger.info("Read " + mQueryTextToId.size() + " queries from " + trecFileName);
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
  
    ArrayList<CandidateEntry> resArr = mTrecRuns.get(queryID);
    if (resArr == null) {
      logger.warn("Ignoring  query " + queryID + 
                 " because trec run contains not entry for cached query ID " + queryID);
      return mEmpty;
    }
    
    int retQty = Math.min(resArr.size(), maxQty);
    CandidateEntry[] results = new CandidateEntry[retQty];
    for (int i = 0; i < retQty; ++i) {
      results[i] = resArr.get(i);
    }
    Arrays.sort(results);
        
    return new CandidateInfo(results.length, results);
  }
  
  private CandidateInfo mEmpty = new CandidateInfo(0, new CandidateEntry[0]);
  private HashMap<String, String>   mQueryTextToId = new HashMap<String, String>();
  private HashMap<String, ArrayList<CandidateEntry>>  mTrecRuns = null;
}
