/*
 *  Copyright 2016 Carnegie Mellon University
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

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.WordEntry;

/**
 * A simple SDM-like similarity, however, it is different in implementation details
 * and is quite close to the one used in (the biggest difference is that we
 * do not compute exact IDFs for word pairs, but rather just average word IDFs):
 * 
 * Boytsov, Leonid, and Anna Belova. 
 * "Evaluating Learning-to-Rank Methods in the Web Track Adhoc Task." TREC. 2011.
 * 
 * @author Leonid Boytsov
 *
 */
public class BM25ClosePairSimilarityQueryNorm extends TFIDFSimilarity implements QueryDocSimilarityFunc {

  public BM25ClosePairSimilarityQueryNorm(float k1, float b, 
                                 int queryWindow, int docWindow, 
                                 ForwardIndex fieldIndex) {
    mBM25_k1 = k1;
    mBM25_b = b;
    
    mQueryWindow = queryWindow;
    mDocWindow = docWindow;
    
    // Division is slow, so it's worth pre-computing the inverse value
    mInvAvgDl = 1.0f/ ((float) fieldIndex.getAvgDocLen());
    mFieldIndex = fieldIndex;
  }
  
  @Override
  protected float computeIDF(float docQty, WordEntry e) {
    float n = e.mWordFreq;
    return (float)Math.log(1 + (docQty - n + 0.5D)/(n + 0.5D));
  }
  
  final float mBM25_k1;
  final float mBM25_b;
  final int  mQueryWindow;
  final int  mDocWindow;
  
  final float mInvAvgDl;
  final ForwardIndex mFieldIndex;
  
  protected long makeKey(int wid1, int wid2) {
    if (wid1 < wid2) {
      return (((long)wid1) << 32) + wid2;
    } else {
      return (((long)wid2) << 32) + wid1;
    }
  }
  
  @Override
  public String getName() {
    return "BM25ClosePair";
  }

  @Override
  public float compute(DocEntry query, DocEntry doc) {
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
          long key = makeKey(wid1, wid2);
          if (!pairQtys.containsKey(key)) {
            qWords1.add(wid1);
            qWords2.add(wid2);
          }
          // Note 1, this would be for zero # of occurrences
          // because zero is the default value we need to put 1 here
          pairQtys.put(makeKey(wid1, wid2), 1); 
        }
      }
    }
    
    // TODO: This nested loop can be a bit slow, in the future some
    //       speed-up might help. However, in a standard scenario
    //       when we read document info from a compressed Lucene field
    //       this seems to be relatively fast.
    for (int i = 0; i < doc.mWordIdSeq.length - 1; ++i) {
      for (int k = i + 1; k < Math.min(doc.mWordIdSeq.length, i + mDocWindow); ++k) {
        int wid1 = Math.min(doc.mWordIdSeq[i], doc.mWordIdSeq[k]);
        int wid2 = Math.max(doc.mWordIdSeq[i], doc.mWordIdSeq[k]);
        if (wid1 >= 0 && wid2 >=0 && wid1 != wid2) {
          long key = makeKey(wid1, wid2);
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
      long key = makeKey(wid1, wid2);
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
    
    return score / getNormIDF(query);
  }
  
  private float getNormIDF(DocEntry query) {
    float normIDF = 0;
    int   queryTermQty = query.mWordIds.length;
    for (int i = 0; i < queryTermQty; ++i) {
      final int queryWordId = query.mWordIds[i];
      if (queryWordId >= 0) {
        float idf = getIDF(mFieldIndex, queryWordId);
        normIDF += idf; // IDF normalization
      }
    }
    return normIDF;
  }

}
