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

import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;

public abstract class CandidateProvider {
  final static Logger logger = LoggerFactory.getLogger(CandidateProvider.class);
  
  public final static String ID_FIELD_NAME    = Const.TAG_DOCNO;
  public final static String QUERY_FIELD_NAME = Const.TEXT_FIELD_NAME;
  
  // If you add a new provider, update CAND_PROVID_DESC below
  public static final String CAND_TYPE_LUCENE      = "lucene";
  public static final String CAND_TYPE_NMSLIB      = "nmslib";
  public static final String CAND_TYPE_TREC_RUNS   = "trec_runs";
 
  public final static String CAND_PROVID_DESC = "candidate record provider type: " + 
      CandidateProvider.CAND_TYPE_LUCENE + ", " + 
      CandidateProvider.CAND_TYPE_NMSLIB + ", " +
      CandidateProvider.CAND_TYPE_TREC_RUNS;
  
  
  /**
   * Create an array of candidate providers. If the provider is thread-safe,
   * only one object will be created.
   * 
   * @param resourceManager	resource
   * @param provType				provider type
   * @param provURI					provider URI (index location or TCP/IP address)
   * @param configName			configuration file name (can be null)
   * @param threadQty				number of threads
   * 
   * @return	an array of providers or null if the provider type is not recognized
   */
  public static CandidateProvider[] createCandProviders(FeatExtrResourceManager resourceManager,
														String provType,
														String provURI,
														String configName,
														int threadQty) throws Exception {
    logger.info(String.format("Provider type: %s URI: %s config file: %s # of threads",
                              provType, provURI, configName != null ? configName : "none", threadQty));
    
  	CandidateProvider[] res = new CandidateProvider[threadQty];
  	
  	CandProvAddConfig addConf = null;
  	
  	if (configName != null) {
  	  addConf = CandProvAddConfig.readConfig(configName, provType);
  	}
  	
  	
    if (provType.equalsIgnoreCase(CandidateProvider.CAND_TYPE_LUCENE)) {
      res[0] = new LuceneCandidateProvider(provURI, addConf);
      for (int ic = 1; ic < threadQty; ++ic) 
        res[ic] = res[0];
    } else if (provType.equalsIgnoreCase(CandidateProvider.CAND_TYPE_TREC_RUNS)) {
      res[0] = new TrecRunCandidateProvider(provURI);
      for (int ic = 1; ic < threadQty; ++ic) 
        res[ic] = res[0];
    } else if (provType.equalsIgnoreCase(CandidateProvider.CAND_TYPE_NMSLIB)) {
      /*
       * NmslibKNNCandidateProvider isn't thread-safe,
       * b/c each instance creates a TCP/IP that isn't supposed to be shared among threads.
       * However, creating one instance of the provider class per thread is totally fine (and is the right way to go). 
       */

      for (int ic = 0; ic < threadQty; ++ic) {
        res[ic] = new NmslibKNNCandidateProvider(provURI, resourceManager, addConf);
      }
             
    } else {
      return null;
    }
  	
  	return res;
  }
  
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
  
  /**
   * Removes stop-word that are often a by-product of tokenization and which
   * isn't always present in standard dictionaries (and thus accidentally
   * added to the index). However, when included into a query, 
   * it drastically can increase retrieval times. We now deprecate this
   * function, because data processing code should take care of removing these words.
   * 
   * @param text
   * @return
   */
  @Deprecated
  public static String removeAddStopwords(String text) {
    if (text != null) {
      text = removeWords(text,  mAddStopWords);
    }
    return text;
  }
  
  static Splitter mSplitOnSpace = Splitter.on(' ');
  static Joiner   mJoinOnSpace  = Joiner.on(' ');
  static String[] mAddStopWords = {Const.PESKY_STOP_WORD};
}


  

