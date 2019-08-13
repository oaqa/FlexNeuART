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
package edu.cmu.lti.oaqa.knn4qa.fwdindx;

import java.util.ArrayList;

/**
 * One document entry in the forward in-memory index.
 * 
 * @author Leonid Boytsov
 *
 */
public class DocEntryParsed {
  public DocEntryParsed(int uniqQty, int [] wordIdSeq, boolean bStoreWordIdSeq) {
    mWordIds = new int [uniqQty];
    mQtys    = new int [uniqQty];
    if (bStoreWordIdSeq)
      mWordIdSeq = wordIdSeq;
    else
      mWordIdSeq = null;
    mDocLen = wordIdSeq.length;
  }

  public DocEntryParsed(ArrayList<Integer> wordIds, 
                  ArrayList<Integer> wordQtys,
                  int                docLen) {
    mWordIds = new int [wordIds.size()];
    mQtys    = new int [wordIds.size()];
    
    for (int i = 0; i < wordIds.size(); ++i) {
      mWordIds[i] = wordIds.get(i);
      mQtys[i] = wordQtys.get(i);
    }
    mWordIdSeq = null;
    mDocLen = docLen;
  }

  
  public DocEntryParsed(ArrayList<Integer> wordIds, 
                  ArrayList<Integer> wordQtys,
                  ArrayList<Integer> wordIdSeq,
                  boolean bStoreWordIdSeq) throws Exception {
    if (wordIds.size() != wordQtys.size()) {
      throw new Exception("Bug: the number of word IDs is not equal to the number of word quantities.");
    }
    mWordIds = new int [wordIds.size()];
    mQtys    = new int [wordIds.size()];
    
    for (int i = 0; i < wordIds.size(); ++i) {
      mWordIds[i] = wordIds.get(i);
      mQtys[i] = wordQtys.get(i);
    }
    
    if (bStoreWordIdSeq) {
      mWordIdSeq = new int [wordIdSeq.size()];
      for (int k = 0; k < wordIdSeq.size(); ++k) {
        mWordIdSeq[k] = wordIdSeq.get(k);
      }
    } else {
      mWordIdSeq = null;
    }
    mDocLen = wordIdSeq.size();
  }
  
  /**
   * Generate a string representation of a document entry:
   * it doesn't add a trailing newline!
   */
  public String toString() {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < mWordIds.length; ++i) {
      if (i > 0) sb.append('\t');
      sb.append(String.format("%d:%d",mWordIds[i] , mQtys[i]));
    }
    sb.append('\n');
    if (mWordIdSeq != null) {
      for (int i = 0; i < mWordIdSeq.length; ++i) {
        if (i > 0) sb.append(' ');
        sb.append(mWordIdSeq[i]);
      }
    } else {
      sb.append(DOCLEN_QTY_PREFIX + mDocLen);
    }
    
    return sb.toString();
  }
  
  public static DocEntryParsed fromString(String text) throws Exception {
    int nli = text.indexOf('\n');
    if (nli < 0) {
      throw new Exception("Invalid document entry, no newline found in:\n" + text);
    }
    String line1 = text.substring(0, nli);

    ArrayList<Integer> wordIds = new ArrayList<Integer>(), wordQtys = new ArrayList<Integer>();
    
    // Some lines can be empty
    if (!line1.isEmpty()) {
      String[] parts = line1.split("\\s+");
      int k = 0;
      for (String s: parts) {
        String [] parts2 = s.split(":");
        ++k;
        if (parts2.length != 2) {
          throw new Exception(
              String.format(
                  "Wrong format: expecting two colon-separated numbers in the substring:" +
                  "'%s' (part # %d)", s, k));          
        }
        
        try {
          int wordId = Integer.parseInt(parts2[0]);
          int wordQty = Integer.parseInt(parts2[1]);
          
          wordIds.add(wordId);
          wordQtys.add(wordQty);
        } catch (NumberFormatException e) {
          throw new Exception(
              String.format(
                      "Invalid document entry format (an ID or count isn't integer), in the substring '%s' (part #%d)",
                      s, k));
        }          
      }
    }
    
    
    String line2 = text.substring(nli + 1);
    
    if (line2.startsWith(DOCLEN_QTY_PREFIX)) {
      final int docLen;
      try {
        docLen = Integer.parseInt(line2.substring(DOCLEN_QTY_PREFIX.length()));
      } catch (NumberFormatException e) {
        throw new Exception("Document length isn't integer");               
      }
      return new DocEntryParsed(wordIds, wordQtys, docLen); 
    } else {
      ArrayList<Integer> wordIdSeq = new ArrayList<Integer>();
      
      if (!line2.isEmpty()) {
        String [] parts = line2.split("\\s+");
        int k = 0;
        for (String s: parts) {
          ++k;
          try {
            int wordId = Integer.parseInt(s);
            wordIdSeq.add(wordId);
          } catch (NumberFormatException e) {
            throw new Exception(
                String.format(
                        "Invalid document entry format (word ID isn't integer), in the substring '%s' (part #%d)",
                        s, k));
          }          
        }
      }
      
      return new DocEntryParsed(wordIds, wordQtys, wordIdSeq, true);
    }    
    
  }
  
  public static final String DOCLEN_QTY_PREFIX = "@ ";
  

  public final int mWordIds[]; // unique word ids
  public final int mQtys[];    // # of word occurrences corresponding to memorized ids
  public final int mWordIdSeq[]; // a sequence of word IDs (can contain repeats), this array CAN BE NULL
  public final int mDocLen; // document length in # of words
}
