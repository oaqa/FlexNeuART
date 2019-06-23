/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;

import edu.cmu.lti.oaqa.solr.SolrServerWrapper;
import edu.cmu.lti.oaqa.solr.UtilConst;

public class SolrCandidateProvider extends CandidateProvider {
  private final static String TEXT_FIELD_NAME   = "Text_bm25";

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  /**
   * Constructor.
   * 
   * @param solrURI               SOLR server URI
   * @param minShouldMatchPCT     A percentage of query words that should appear in a document
   * @throws Exception
   */
  public SolrCandidateProvider(String solrURI, int minShouldMatchPCT) throws Exception {
    mSolr = new SolrServerWrapper(solrURI);
    mMatchPCT = minShouldMatchPCT;
  }
  
  // The function getCandidates is thread-safe.
  @Override
  public boolean isThreadSafe() { return true; }   
  
  @Override
  public CandidateInfo getCandidates(int queryNum, 
                                Map<String, String> queryData, 
                                int maxQty) throws Exception {
    ArrayList<CandidateEntry> resArr = new ArrayList<CandidateEntry>();
    
    String queryID = queryData.get(ID_FIELD_NAME);
    if (null == queryID) {
      throw new Exception(
          String.format("Query id (%s) is undefined for query # %d",
                        ID_FIELD_NAME, queryNum));
    }        
    
    String text = queryData.get(TEXT_FIELD_NAME);
    if (null == text) {
      throw new Exception(
          String.format("Query (%s) is undefined for query # %d",
                        TEXT_FIELD_NAME, queryNum));
    }
    
    String query = text;
    
    if (mMatchPCT > Float.MIN_NORMAL) {
      int qty = text.split(" +").length;
      // Rounding one half up
      int matchNum = (qty * mMatchPCT + 50) / 100;
      
      query =  String.format("_query_: \"{!edismax mm=%d} %s \"",
                                matchNum,   
                                text);      
    }
    
    //System.out.println("SOLR Query: " + query);

    List<String> fieldList = new ArrayList<String>();
    fieldList.add(ID_FIELD_NAME);
    fieldList.add(UtilConst.SCORE_FIELD);    
    
    SolrDocumentList solrRes = mSolr.runQuery(query, TEXT_FIELD_NAME, 
                                              fieldList, 
                                              null, maxQty);    
    
    for (SolrDocument doc : solrRes) {
      String id  = (String)doc.getFieldValue(ID_FIELD_NAME);
      float  score = (Float)doc.getFieldValue(UtilConst.SCORE_FIELD);
      resArr.add(new CandidateEntry(id, score));
    }
    
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
    Arrays.sort(results);
        
    return new CandidateInfo((int)solrRes.getNumFound(), results);
  }
  
  private final SolrServerWrapper mSolr;
  private final int               mMatchPCT;
}
