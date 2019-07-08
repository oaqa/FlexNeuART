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
package edu.cmu.lti.oaqa.knn4qa.fwdindx;

/**
 * One word entry in the dictionary of the forward in-memory index.
 * 
 * @author Leonid Boytsov
 *
 */
public class WordEntry {
  public int mWordId = 0;
  public int mWordFreq = 0;
  
  WordEntry(int wordId) {
    mWordId = wordId;
  }
  WordEntry(int wordId, int docQty) {
    mWordId = wordId;
    mWordFreq = docQty;
  }
}
