package edu.cmu.lti.oaqa.flexneuart.simil_func;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;

/**
 * A re-implementation of the Lucene/SOLR BM25 similarity,
 * which is normalized using the sum of query term IDFs. 
 * 
 * @author Leonid Boytsov
 *
 */
public class BM25SimilarityLuceneNorm extends BM25SimilarityLucene {
  
  public BM25SimilarityLuceneNorm(float k1, float b, ForwardIndex fieldIndex) {
    super(k1, b, fieldIndex);
  }
  
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
    float score = super.compute(query, doc);

    float normIDF = getNormIDF(query);

    if (normIDF > 0) score /= normIDF;
    
    return score;
  }
  
  private float getNormIDF(DocEntryParsed query) {
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
  
  @Override
  public TrulySparseVector getSparseVector(DocEntryParsed e, boolean isQuery) {
    boolean shareIDF = true;
    TrulySparseVector res = getBM25SparseVectorNoNorm(e, isQuery, shareIDF);
    
    if (isQuery) {
      float normIDF = getNormIDF(e);
      if (normIDF > 0) {
        float inv = 1.0f / normIDF;
        for (int i = 0; i < res.mIDs.length; ++i) {
          res.mVals[i] *= inv;
        }
      }
    }
    
    return res;
  }
  
  @Override
  public String getName() {
    return "BM25 query-norm";
  }  
}
