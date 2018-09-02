package edu.cmu.lti.oaqa.knn4qa.simil;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.WordEntry;

/**
 * Just the cosine similarity between TF*IDF vectors (using BM25 IDF).
 * 
 * @author Leonid Boytsov
 *
 */
public class CosineTextSimilarity extends TFIDFSimilarity {
  public CosineTextSimilarity(InMemForwardIndex fieldIndex) {
    mFieldIndex = fieldIndex;
  }
  
  @Override
  protected float computeIDF(float docQty, WordEntry e) {
    float n = e.mWordFreq;
    return (float)Math.log(1 + (docQty - n + 0.5D)/(n + 0.5D));
  }

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
    
    int   queryTermQty = query.mWordIds.length;

    float normQuery =0;    
    for (int iQuery = 0; iQuery < queryTermQty; ++iQuery) {
      final int queryWordId = query.mWordIds[iQuery];
      if (queryWordId >= 0) {
        float idf = getIDF(mFieldIndex, queryWordId);
        float w = query.mQtys[iQuery]*idf;
        normQuery += w * w; 
      }
    }
    
    int   docTermQty = doc.mWordIds.length;
    
    float normDoc = 0;
    for (int iDoc = 0; iDoc < docTermQty; ++iDoc) {
      final int docWordId   = doc.mWordIds[iDoc];
      // docWordId >= 0 should always be non-negative (unlike queryWordId, which can be -1 for OOV words 
      float idf = getIDF(mFieldIndex, docWordId);
      float w = doc.mQtys[iDoc]*idf;
      normDoc += w * w;
    }
    
    int   iQuery = 0, iDoc = 0;
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else { 
        // Here queryWordId == docWordId
        float idf = getIDF(mFieldIndex, docWordId);
        score +=  query.mQtys[iQuery] * idf * doc.mQtys[iDoc] * idf;
        
        ++iQuery; ++iDoc;
      }
    }
    
    return score /= Math.sqrt(Math.max(1e-6, normQuery * normDoc));
  }

  @Override
  public String getName() {
    return "Cosine text";
  }
}
