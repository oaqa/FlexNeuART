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
package edu.cmu.lti.oaqa.knn4qa.giza;

import java.util.*;
import java.io.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VocabularyFilter;

/**
 * 
 * A helper class to read vocabulary files produced by Giza or Giza++. In
 * addition to mapping words to respective IDs, it also computes word
 * probabilities. Words are treated as is, i.e., no lowercasing is done.
 * 
 * @author Leonid Boytsov
 *
 */
public class GizaVocabularyReader {
  private static final Logger logger = LoggerFactory.getLogger(GizaVocabularyReader.class);
  
  /**
   * A constructor that reads the file name, builds the word-to-ID map,
   * and computes log probabilities. One can specify a "filter" vocabulary
   * to load only a subset of all words.
   * 
   * @param     fileName    main vocabulary file name.
   * @param     filter      already loaded filter vocabulary, can be null.
   * @throws    Exception 
   */
  public GizaVocabularyReader(String fileName, VocabularyFilter filter) throws Exception {
    int         qty = 0;
    double      totOccQty = 0;
            
    {
      // Pass 1: compute the # of records and the total # of occurrences
      BufferedReader fr = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fileName)));
      
      String line = null;
      
      while ((line = fr.readLine()) != null) {
        // Skip empty lines
        line = line.trim(); if (line.isEmpty()) continue;
        
        GizaVocRec rec = new GizaVocRec(line);
        ++qty; totOccQty += rec.mQty;
      }
      
      mProb    = new double[qty];
      mId      = new int[qty];
      mWords   = new String[qty];
      
      fr.close();
    }
    
    {
      // Pass 2: re-read the file and compute probabilities/
      BufferedReader fr = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fileName)));
      
      String line = null;
      
      for (int pos = 0; (line = fr.readLine()) != null; ) {
        // Skip empty lines
        line = line.trim(); if (line.isEmpty()) continue;
        
        GizaVocRec rec = new GizaVocRec(line);
        if (mWord2InternIdMap.get(rec.mWord) != null) {
          throw new Exception("Repeating word: '" + rec.mWord+ "' in file: '" + fileName + "'");
        }
        if (mId2InternIdMap.get(rec.mId) != null) {
          throw new Exception("Repeating ID: '" + rec.mId+ "' in file: '" + fileName + "'");
        }
        
        if (filter == null || filter.checkWord(rec.mWord)) {
          mWord2InternIdMap.put(rec.mWord, pos);
          mId2InternIdMap.put(rec.mId, pos);
          
          mProb[pos]      = ((double)rec.mQty)/ totOccQty;
          mId[pos]        = rec.mId;
          mWords[pos]     = rec.mWord;
        }
        ++pos;
      }
      
      fr.close();
    }
    
    logger.info("Read the vocabulary from '" + fileName + "'");
  }
  
  /**
   * Obtain word ID.
   * 
   * @param word
   * @return a word ID or NULL, if the word wasn't in the Giza vocabulary file.
   */
  public Integer getWordId(String word) {
    Integer pos = mWord2InternIdMap.get(word);
    if (pos == null) return null;
    return mId[pos];
  }

  /**
   * Obtain a probability of a word.
   * 
   * @param word
   * @return a word probability or NULL, if the word isn't in the Giza vocabulary file.
   */  
  public double getWordProb(String word) {
    Integer pos = mWord2InternIdMap.get(word);
    if (pos == null) return 0.0;
    return mProb[pos];
  }
  
  
  public String getWord(int wordId) {
    Integer pos = mId2InternIdMap.get(wordId);
    if (pos == null) return null;
    return mWords[pos]; 
  }
  
  private HashMap<String, Integer>  mWord2InternIdMap = new HashMap<String, Integer>();
  private HashMap<Integer,Integer>  mId2InternIdMap = new HashMap<Integer, Integer>();
  private String[]                  mWords = null;
  private double[]                  mProb = null;
  private int[]                     mId = null;
}
