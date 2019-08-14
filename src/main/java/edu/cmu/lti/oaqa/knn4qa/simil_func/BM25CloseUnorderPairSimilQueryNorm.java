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
package edu.cmu.lti.oaqa.knn4qa.simil_func;

import net.openhft.koloboke.collect.map.hash.HashLongIntMap;
import net.openhft.koloboke.collect.map.hash.HashLongIntMaps;

import java.util.ArrayList;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;

public class BM25CloseUnorderPairSimilQueryNorm extends BM25ClosePairSimilarityQueryNormBase implements QueryDocSimilarityFunc {

  public BM25CloseUnorderPairSimilQueryNorm(float k1, float b, 
                                         int queryWindow, int docWindow, 
                                         ForwardIndex fieldIndex) {
    super(k1, b, queryWindow, docWindow, fieldIndex);
  }
  
  @Override
  public String getName() {
    return "BM25CloseUnorderPair";
  }
  

  private long makeUnorderedKey(int wid1, int wid2) {
    if (wid1 < wid2) {
      return (((long)wid1) << 32) + wid2;
    } else {
      return (((long)wid2) << 32) + wid1;
    }
  }

  @Override
  public float compute(DocEntryParsed query, DocEntryParsed doc) {
    HashLongIntMap pairQtys = HashLongIntMaps.newMutableMap();
    ArrayList<Integer> qWords1 = new ArrayList<Integer>();
    ArrayList<Integer> qWords2 = new ArrayList<Integer>();
    
    if (doc.mWordIdSeq == null) {
      throw new RuntimeException("This similarity requires a positional forward index!");
    }
    
    for (int i = 0; i < query.mWordIdSeq.length - 1; ++i) {
      for (int k = i + 1; k < Math.min(query.mWordIdSeq.length, i + mQueryWindow); ++k) {
        int wid1 = Math.min(query.mWordIdSeq[i], query.mWordIdSeq[k]);
        int wid2 = Math.max(query.mWordIdSeq[i], query.mWordIdSeq[k]);
        if (wid1 >= 0 && wid2 >=0 && wid1 != wid2) {
          long key = makeUnorderedKey(wid1, wid2);
          if (!pairQtys.containsKey(key)) {
            qWords1.add(wid1);
            qWords2.add(wid2);
          }
          // Note one, this denotes zero # of occurrences:
          // because zero is the default hash value we need to put 1 here
          pairQtys.put(key, 1); 
        }
      }
    }
    
    // TODO: This nested loop can be a bit slow, in the future some speed-up might help. 
    for (int i = 0; i < doc.mWordIdSeq.length - 1; ++i) {
      for (int k = i + 1; k < Math.min(doc.mWordIdSeq.length, i + mDocWindow); ++k) {
        int wid1 = Math.min(doc.mWordIdSeq[i], doc.mWordIdSeq[k]);
        int wid2 = Math.max(doc.mWordIdSeq[i], doc.mWordIdSeq[k]);
        if (wid1 >= 0 && wid2 >=0 && wid1 != wid2) {
          long key = makeUnorderedKey(wid1, wid2);
          int val = pairQtys.get(key);
          
          // Increase counts only for query word pairs
          if (pairQtys.defaultValue() != val) {
            pairQtys.put(key,  val + 1);
          }
        }
      }
    }
    
    float score = 0;
    float docLen = doc.mDocLen;
    
    for (int i = 0; i < qWords1.size(); i++) {
      int wid1 = qWords1.get(i);
      int wid2 = qWords2.get(i);
      long key = makeUnorderedKey(wid1, wid2);
      int val = pairQtys.get(key);
      if (val > 1) {
        float idf1 = getIDF(mFieldIndex, wid1);
        float idf2 = getIDF(mFieldIndex, wid2);
        float tf = val - 1;
        float normTf = (tf * (mBM25_k1 + 1)) / ( tf + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));
        
        // We simply take an average IDF
        score += normTf * 0.5 * (idf1 + idf2);
      }
    }
    
    float normIDF = getNormIDF(query);
    if (normIDF > 0) {
      score /= normIDF;
    }
    return score;
  }

}
