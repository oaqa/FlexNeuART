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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.WordEntry;

/**
 * A re-implementation of the Lucene BM25 similarity. 
 * 
 * @author Leonid Boytsov
 *
 */
public class BM25SimilarityLucene extends TFIDFSimilarity {
  public static final float DEFAULT_BM25_K1 = 1.2f;
  public static final float DEFAULT_BM25_B = 0.75f; 
  
  public BM25SimilarityLucene(float k1, float b, ForwardIndex fieldIndex) {
    mBM25_k1 = k1;
    mBM25_b = b;
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
  
  final float mInvAvgDl;
  final ForwardIndex mFieldIndex;
  
  /**
   * Computes the similarity between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  @Override
  public float compute(DocEntryParsed query, DocEntryParsed doc) {
    float score = 0;
    
    int   docTermQty = doc.mWordIds.length;
    int   queryTermQty = query.mWordIds.length;
    
    int   iQuery = 0, iDoc = 0;
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else {
        
        // This computation can be made a bit efficient by keeping
        // document length and the word id in a local variable,
        // but the gains would be small and this way we provide
        // a function to compute a document term weight
        float termDocScore = getDocTermScore(doc, iDoc); 
        
        score += termDocScore * query.mQtys[iQuery];            
        ++iQuery; ++iDoc;
      }
    }
    
    return score;
  }

  /**
   * Computes a BM25 weight of a given word in a document.
   * 
   * @param doc     document entry reference
   * @param iDoc    the zero-based position of the word in 
   *                the BOW list of document words.
   * @return
   */
  public float getDocTermScore(DocEntryParsed doc, int iDoc) {
    float tf = doc.mQtys[iDoc];
    float docLen = doc.mDocLen;
    
    // Normalized term frequency    
    float normTf = (tf * (mBM25_k1 + 1)) / ( tf + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));
    float idf = getIDF(mFieldIndex, doc.mWordIds[iDoc]) ;
    float termDocScore = idf * normTf;
    return termDocScore;
  }

  
  @Override
  public String getName() {
    return "BM25";
  }

  public float [] computeEmbed(float[][] distMatrixCosine, DocEntryParsed query, DocEntryParsed doc) {
    float docLen = doc.mDocLen;
    float scores[] = new float[2];
    
    int queryQty = query.mWordIds.length;
    int docQty = doc.mWordIds.length;
    if (queryQty == 0 || docQty == 0) return scores;

    /*
    float docIDF[] = new float[docQty];
    for (int id = 0; id < docQty; ++id) {
      docIDF[id] = getIDF(mFieldIndex, doc.mWordIds[id]);
    }
    */
    
    for (int iq = 0; iq < queryQty; ++iq) {
      int qWordId = query.mWordIds[iq];
      if (qWordId < 0) continue;

      float docTF1 = 0, docTF2 = 0;
      float queryIDF = getIDF(mFieldIndex, qWordId);

      for (int id = 0; id < docQty; ++id) {
        float distMatrScore = distMatrixCosine[iq][id];
        if (Float.isInfinite(distMatrScore)) continue;

        float tf  = doc.mQtys[id];
        float similScore = 0.5f*(2 - distMatrScore);

        if (similScore >= 0.75f) {
          float tfs = tf * similScore;
          docTF1 += tfs;
          if (tfs > docTF2) {
            docTF2 = tfs;
          }
        }
      }

      float normTF1 = (docTF1 * (mBM25_k1 + 1)) / ( docTF1 + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));
      float normTF2 = (docTF2 * (mBM25_k1 + 1)) / ( docTF2 + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));

      float queryTF = query.mQtys[iq];

      scores[0] += queryIDF * queryTF * normTF1;
      if (Float.isNaN(scores[0])) {
        throw new RuntimeException("Obtained the NaN score[0]!");
      }
      scores[1] += queryIDF * queryTF * normTF2;
      if (Float.isNaN(scores[1])) {
        throw new RuntimeException("Obtained the NaN score[0]!");
      }      
    }
    return scores;
  }  

  @Override
  public TrulySparseVector getSparseVector(DocEntryParsed e, boolean isQuery) {
    return getBM25SparseVectorNoNorm(e, isQuery, true /* shareIDF */);
  }

  protected TrulySparseVector getBM25SparseVectorNoNorm(DocEntryParsed e, boolean isQuery, boolean shareIDF) {
    int qty = 0;
    for (int wid : e.mWordIds)
      if (wid >= 0) qty++;
    TrulySparseVector res = new TrulySparseVector(qty);
    
    float docLen = e.mDocLen;
    
    for (int i = 0, id=0; i < e.mWordIds.length; ++i) {
      int wordId = e.mWordIds[i];
      if (wordId < 0) continue;
      
      
      float IDF = getIDF(mFieldIndex, wordId);
      float tf = e.mQtys[i];
      
      res.mIDs[id] = wordId;
      if (isQuery) {
        res.mVals[id]=  tf * (shareIDF ? (float) Math.sqrt(IDF) : 1); 
      } else {
        float tfScaled = (tf * (mBM25_k1 + 1)) / ( tf + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));
        res.mVals[id]=  (shareIDF ? (float) Math.sqrt(IDF) : IDF) * tfScaled;
      }
      id++;
    }
    
    return res;
  }
  
}
