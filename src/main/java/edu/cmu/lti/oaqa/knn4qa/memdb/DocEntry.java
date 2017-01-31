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
package edu.cmu.lti.oaqa.knn4qa.memdb;

import java.util.ArrayList;

/**
 * One document entry in the forward in-memory index.
 * 
 * @author Leonid Boytsov
 *
 */
public class DocEntry {
  public DocEntry(int uniqQty, int [] wordIdSeq, boolean bStoreWordIdSeq) {
    mWordIds = new int [uniqQty];
    mQtys    = new int [uniqQty];
    if (bStoreWordIdSeq)
      mWordIdSeq = wordIdSeq;
    else
      mWordIdSeq = null;
    mDocLen = wordIdSeq.length;
  }

  public DocEntry(ArrayList<Integer> wordIds, 
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

  
  public DocEntry(ArrayList<Integer> wordIds, 
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

  public final int mWordIds[]; // unique word ids
  public final int mQtys[];    // # of word occurrences corresponding to memorized ids
  public final int mWordIdSeq[]; // a sequence of word IDs (can contain repeats), this array CAN BE NULL
  public final int mDocLen; // document length in # of words
}
