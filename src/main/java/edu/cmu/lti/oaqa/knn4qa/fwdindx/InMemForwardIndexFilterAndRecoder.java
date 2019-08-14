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

import edu.cmu.lti.oaqa.knn4qa.fwdindx.*;

public class InMemForwardIndexFilterAndRecoder extends
    VocabularyFilterAndRecoder {
  private ForwardIndex mIndex;

  public InMemForwardIndexFilterAndRecoder(ForwardIndex index) {
    mIndex = index;
  }
  
  @Override
  public boolean checkWord(String word) {
    WordEntry e = mIndex.getWordEntry(word);
    return e != null;
  }

  @Override
  public Integer getWordId(String word) {
    WordEntry e = mIndex.getWordEntry(word);
    if (e != null) return e.mWordId;
    return null;
  }
  
}
