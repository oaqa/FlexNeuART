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
package edu.cmu.lti.oaqa.flexneuart.giza;

import java.util.*;
import java.io.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.VocabularyFilter;

/**
 * 
 * A helper class to read both the source and the target vocabulary files produced by Giza or Giza++. 
 * It reads only the words and stores them in a hash set to quickly verify if a given word belongs
 * to either of the input vocabulary files.
 * 
 * @author Leonid Boytsov
 *
 */
public class GizaVocabularyFilterStand extends VocabularyFilter {
  private static final Logger logger = LoggerFactory.getLogger(GizaVocabularyFilterStand.class);
  
  /**
   * Constructor.
   * 
   * @param     fileNames vocabulary files (the order doesn't matter)
   *
   * @throws    Exception 
   */
  public GizaVocabularyFilterStand(String ... fileNames) throws Exception {
    for (String fn: fileNames)
      readVocFile(fn);
  }

  /**
   *  Read one vocabulary file.
   *
   * @param   fileName vocabulary file
   * @throws  Exception
   */
  private void readVocFile(String fileName) throws Exception {
    int         qty = 0;
    double      totOccQty = 0;
            
    {
      // Pass 1: compute the # of records and the total # of occurrences
      BufferedReader fr = new BufferedReader(new FileReader(fileName));
      
      String line = null;
      
      while ((line = fr.readLine()) != null) {
        // Skip empty lines
        line = line.trim(); if (line.isEmpty()) continue;
        
        GizaVocRec rec = new GizaVocRec(line);
        mWords.add(rec.mWord);
        ++qty; totOccQty += rec.mQty;
      }

      fr.close();
    }
    logger.info("Read the vocabulary from '" + fileName + "'");
  }
  
  /**
   * Checks if the word is present.
   * 
   * @param word
   * @return true/false
   */
  @Override
  public boolean checkWord(String word) {
    return mWords.contains(word);
  }
  
  private HashSet<String>  mWords = new HashSet<String>();
}
