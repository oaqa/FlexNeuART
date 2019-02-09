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

import java.util.*;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.annographix.solr.SolrRes;
import edu.cmu.lti.oaqa.annographix.solr.UtilConst;

public abstract class CandidateProvider {
  public final static String ID_FIELD_NAME     = UtilConst.TAG_DOCNO;

  public final static String TEXT_FIELD_NAME = "text";
  public final static String TEXT_UNLEMM_FIELD_NAME = "text_unlemm";
      
  public static final String CAND_TYPE_QRELS       = "qrels";
  public static final String CAND_TYPE_SOLR        = "solr";
  public static final String CAND_TYPE_LUCENE      = "lucene";
  public static final String CAND_TYPE_LUCENE_GIZA = "lucene_giza";
  public static final String CAND_TYPE_GALAGO      = "galago";
  public static final String CAND_TYPE_NMSLIB      = "nmslib";
  

  public final static String CAND_PROVID_DESC = "candidate record provider type: " + 
      CandidateProvider.CAND_TYPE_LUCENE + ", " + 
      CandidateProvider.CAND_TYPE_QRELS + ", " + 
      CandidateProvider.CAND_TYPE_SOLR + ", " + 
      CandidateProvider.CAND_TYPE_GALAGO + "," + 
      CandidateProvider.CAND_TYPE_NMSLIB;
  
  /**
   * @return  true if {@link #getCandidates(int, Map, int)} can be called by 
   *               several threads simultaneously. 
   *           
   */
  public abstract boolean isThreadSafe();
  
  public abstract String getName();
  
  /**
   * Parses a QREL label and checks if it is a non-negative integer
   * 
   * @param label       the string label (can be null).
 
   *         the label is null.
   * @throws Exception throws an exception if the label is not numeric
   */
  public static int parseRelevLabel(String label) throws Exception {
    if (null == label) return 0; // no relevance entry => not relevant
    int relVal = 0;
    try {
      relVal = Integer.parseInt(label);
    } catch (NumberFormatException e) {
      throw new Exception("Label '" + label + "' is not numeric!");
    }
    if (relVal < 0) {
      throw new Exception("Encountered a negative relevance label: " + label);
    }
    return relVal ;
  }
  
  
  /**
   * Return a <b>sorted</b> list of candidate records with respective similarity scores +
   * the total number of entries found.
   * 
   * @param     queryNum     an ordinal query number (for debugging purposes).
   * @param     queryData    several pieces of input data, one is typically a bag-of-words query.
   * @param     maxQty       a maximum number of candidate records to return.
   * @return    an array of candidate records (doc ids + scores, no document text) + 
   *            the total number of entries found.
   */
  abstract public CandidateInfo getCandidates(int queryNum, 
                                    Map<String, String> queryData, 
                                    int maxQty)  throws Exception;
  
  /**
   * Removes words from the list (case insensitive matching), assumes that the query words are separated by spaces. 
   * It's will be inefficient for long word lists, though.
   * 
   * @param origQuery
   * @param wordList
   * @return
   */
  public static String removeWords(String origQuery, String[] wordList) {
     ArrayList<String> res = new ArrayList<String>();
     for (String s : mSplitOnSpace.split(origQuery)) {
       boolean f = true;
       for (int k = 0; k < wordList.length; ++k)
         if (s.equalsIgnoreCase(wordList[k])) { f = false; break; }
       if (f) res.add(s);      
     }
     return mJoinOnSpace.join(res);
  }
  static Splitter mSplitOnSpace = Splitter.on(' ');
  static Joiner   mJoinOnSpace  = Joiner.on(' ');
}
