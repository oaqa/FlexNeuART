package edu.cmu.lti.oaqa.knn4qa.simil;

import java.util.*;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.WordEntry;

/**
 * A re-implementation of the Lucene/SOLR BM25 similarity,
 * which is normalized using the sum of query term IDFs. 
 *
 * <p>Unlike
 * the original implementation, though, we don't rely on a coarse
 * version of the document normalization factor. Our approach
 * might be a tad slower, but
 * (1) it's easier to implement;
 * (2) there is a small (about 1%) increase in accuracy. 
 * Note that IDF values are cached (without evicting from the cache).  
 * </p>
 * 
 * @author Leonid Boytsov
 *
 */
public class BM25SimilarityLuceneNorm extends TFIDFSimilarity {
  public BM25SimilarityLuceneNorm(float k1, float b, InMemForwardIndex fieldIndex) {
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
  final InMemForwardIndex mFieldIndex;
  
  /**
   * Computes the similarity between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  @Override
  public float compute(DocEntry query, DocEntry doc) {
    float score = 0;
    
    int   docTermQty = doc.mWordIds.length;
    int   queryTermQty = query.mWordIds.length;

    float normIDF = 0;
    for (int i = 0; i < queryTermQty; ++i) {
      final int queryWordId = query.mWordIds[i];
      if (queryWordId >= 0) {
        float idf = getIDF(mFieldIndex, queryWordId);
        normIDF += idf; // IDF normalization
      }
    }
    
    int   iQuery = 0, iDoc = 0;
    
    float docLen = doc.mDocLen;
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else {
        float tf = doc.mQtys[iDoc];
        
        float normTf = (tf * (mBM25_k1 + 1)) / ( tf + mBM25_k1 * (1 - mBM25_b + mBM25_b * docLen * mInvAvgDl));
        
        float idf = getIDF(mFieldIndex, query.mWordIds[iQuery]);
        score +=  idf * // IDF 
                  query.mQtys[iQuery] *           // query frequency
                  normTf;                         // Normalized term frequency        
        ++iQuery; ++iDoc;
      }
    }

    if (normIDF > 0) score /= normIDF;
    
    return score;
  }
  
  @Override
  public String getName() {
    return "BM25";
  }  
}
