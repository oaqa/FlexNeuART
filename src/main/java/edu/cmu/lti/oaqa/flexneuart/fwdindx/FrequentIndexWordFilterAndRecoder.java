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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex.ForwardIndexFieldType;

class WordEntryStr {
  public final int mWordId;
  public final int mWordFreq;
  public final String mWord;
  
  WordEntryStr(int wordId, int docQty, String word) {
    mWordId = wordId;
    mWordFreq = docQty;
    mWord = word;
  }
}

class CompareWordEntryByFreqDesc implements Comparator<WordEntryStr> {

  @Override
  public int compare(WordEntryStr o1, WordEntryStr o2) {
    if (o1.mWordFreq != o2.mWordFreq) return o2.mWordFreq - o1.mWordFreq;
    return o1.mWordId - o2.mWordId;  
  }
}



/**
 * A class that creates a dictionary filters based on maxWordQty most common
 * words.
 * 
 * @param fileName        the forward-index file name.
 * @param maxWordQty   the maximum number of words to use,
 *                        if the numWordQty-th most frequent words have frequency
 *                        QTY, we include all words that appear at least QTY times. 
 * 
 * @author Leonid Boytsov
 *
 */
public class FrequentIndexWordFilterAndRecoder extends VocabularyFilterAndRecoder {
  public FrequentIndexWordFilterAndRecoder(String fileName, int maxWordQty) throws Exception {
    ArrayList<WordEntryStr> words = new ArrayList<WordEntryStr>();
    
    if (maxWordQty <= 0) 
      throw new Exception("The maximum # of words should be a positive integer!");
    
    BufferedReader  inp = null;
    
    try {
      inp = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
      
      String indexFieldTypeStr = inp.readLine();
      
      ForwardIndexFieldType indexFieldType = ForwardIndex.getIndexFieldType(indexFieldTypeStr);
      
      if (indexFieldType != ForwardIndexFieldType.parsedBOW && indexFieldType != ForwardIndexFieldType.parsedText) {
        throw 
        new Exception("The filter/recorder requires a parsed forward file, but " + fileName + " has type: " + indexFieldType);
      }

      String meta = inp.readLine();

      if (null == meta)
        throw new Exception(
            String.format(
                    "Can't read meta information: the file '%s' may have been truncated.",
                    fileName));

      String parts[] = meta.split("\\s+");
      if (parts.length != 2)
        throw new Exception(
            String.format(
                    "Wrong format, file '%s': meta-information (first line) should contain exactly two space-separated numbers.",
                    fileName));
      

      String line = inp.readLine();
      if (line == null || !line.isEmpty()) {
        String.format(
                "Can't read an empty line after meta information: the file '%s' may have been truncated.",
                fileName);
      }
      
      // First read the dictionary
      int lineNum = 3;
      line = inp.readLine();
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        parts = line.split("\\t");
        if (parts.length != 2) {
          throw new Exception(
              String.format(
                      "Invalid dictionary format (should be two tab-separated parts), line %d, file %s",
                      lineNum, fileName));
        }
        String w = parts[0];
        String[] partSuff = parts[1].split(":");
        if (partSuff.length != 2) {
          throw new Exception(
              String.format(
                      "Invalid dictionary entry format (should end with two colon separated integers), line %d, file %s",
                      lineNum, fileName));
        }

        int wordId = -1;
        int docQty = -1;

        try {
          wordId = Integer.parseInt(partSuff[0]);
          docQty = Integer.parseInt(partSuff[1]);
        } catch (NumberFormatException e) {
          throw new Exception(
              String.format(
                      "Invalid dictionary entry format (an ID or count isn't integer), line %d, file %s",
                      lineNum, fileName));
        }
        if (wordId < ForwardIndex.MIN_WORD_ID) {
          throw new Exception(
                      String.format("Inconsistent data, wordId %d is too small, should be>= %d", 
                                    wordId, ForwardIndex.MIN_WORD_ID));
        }
        words.add(new WordEntryStr(wordId, docQty, w));
      }
      
      if (line == null)
        throw new Exception(
            String.format(
                    "Can't read an empty line (line number %d): the file '%s' may have been truncated.",
                    lineNum, fileName));
            
      
      System.out.println("Finished loading dictionary from file: " + fileName + " # of entries: " + words.size());
      
      int minFreq = 0;
      
      if (words.size() >= maxWordQty) {
        words.sort(new CompareWordEntryByFreqDesc());
        
        minFreq = words.get(maxWordQty - 1).mWordFreq;
      }
      
      for (WordEntryStr e: words) 
      if (e.mWordFreq >= minFreq) {
        mStr2Int.put(e.mWord, e.mWordId);
        mInt.add(e.mWordId);
      }
      
      System.out.println("Retained " + mStr2Int.size() + " words for filtering!");      
    } finally {    
      if (null != inp) inp.close();
    }    
  }
  
  
  @Override
  public boolean checkWord(String word) {
    return mStr2Int.containsKey(word);
  }
  

  @Override
  public Integer getWordId(String word) {
    return mStr2Int.get(word);
  }
  
  public boolean checkWordId(int id) {
    return mInt.contains(id);
  }


  HashMap<String, Integer>    mStr2Int = new HashMap<String, Integer>();
  HashSet<Integer>            mInt = new HashSet<Integer>();
  
  public static void main(String args[]) throws NumberFormatException, Exception {
    FrequentIndexWordFilterAndRecoder flt =
        new FrequentIndexWordFilterAndRecoder(args[0], Integer.parseInt(args[1]));
  }
}
