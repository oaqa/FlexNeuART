
package edu.cmu.lti.oaqa.knn4qa.simil_func;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.WordEntry;

/**
 * <p>A simple SDM-like similarity, however, it is different in implementation details
 * and is quite close to the one used in (the biggest difference is that we
 * do not compute exact IDFs for word pairs, but rather just average word IDFs):</p>
 * 
 * <p>Boytsov, Leonid, and Anna Belova. 
 * "Evaluating Learning-to-Rank Methods in the Web Track Adhoc Task." TREC. 2011.</p>
 * 
 * <p>Note that this a base class with two sub-classes (ordered and ordered pairs)</p>
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class BM25ClosePairSimilarityQueryNormBase extends TFIDFSimilarity {

  protected final float mBM25_k1;
  protected final float mBM25_b;
  protected final int mQueryWindow;
  protected final int mDocWindow;
  protected final float mInvAvgDl;
  protected final ForwardIndex mFieldIndex;

  public BM25ClosePairSimilarityQueryNormBase(float k1, float b, 
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

  protected float getNormIDF(DocEntryParsed query) {
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