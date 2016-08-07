package edu.cmu.lti.oaqa.knn4qa.simil;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.WordEntry;

/**
 * A re-implementation of the default Lucene/SOLR BM25 similarity,
 * which is <b>normalized</b> using the sum of query squared IDFs. 
 *
 * <p>Unlike the original implementation, though, we don't rely on a coarse
 * version of the document normalization factor. 
 * Our approach is easier to implement.</p>
 * 
 * @author Leonid Boytsov
 *
 */
public class DefaultSimilarityLuceneNorm extends QueryDocSimilarity {
  public DefaultSimilarityLuceneNorm(InMemForwardIndex fieldIndex) {
    mFieldIndex = fieldIndex;
  }
  
  @Override
  protected float computeIDF(float docQty, WordEntry e) {
    float n = e.mWordFreq;
    return (float)(Math.log(docQty/(double)(n + 1)) + 1.0);
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
    
    int   docTermQty = doc.mWordIds.length;
    int   queryTermQty = query.mWordIds.length;

    float normIDF=0;
    for (int i = 0; i < queryTermQty; ++i) {
      final int queryWordId = query.mWordIds[i];
      if (queryWordId >= 0) {
        float idf = getIDF(mFieldIndex, query.mWordIds[i]);
        normIDF += idf*idf; // IDF normalization
      }
    }
    
    int   iQuery = 0, iDoc = 0;
    
    float docLen = doc.mWordIdSeq.length;
    
//    float queryNorm = 0;
    float lengthNorm = docLen > 0 ? ((float) (1.0 / Math.sqrt(docLen))) : 0;
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else {
        float tf = (float)Math.sqrt(doc.mQtys[iDoc]);
        
        float idf = getIDF(mFieldIndex, query.mWordIds[iQuery]);
        float idfSquared = idf * idf;
        
//        System.out.println(String.format("## Word %s sqrt(tf)=%f idf=%f", 
//                                        mFieldIndex.getWord(query.mWordIds[iQuery]), tf, idf));
        
// Contrary to what docs say: It looks like Lucene actually doesn't use this query normalizer        
//        queryNorm += idfSquared;
        
        score +=  query.mQtys[iQuery] *           // query frequency
                  tf * idfSquared;
        
        ++iQuery; ++iDoc;
      }
    }
    
//    queryNorm = (float)Math.sqrt(queryNorm);
    
//    System.out.println(String.format("## queryNorm=%f lengthNorm=%f", queryNorm, lengthNorm));
    
    score *= lengthNorm;
    
// Contrary to what the docs say: It looks like Lucene actually doesn't use this query normalizer    
//    if (queryNorm > 0)
//      score /= queryNorm;

    if (normIDF > 0) score /= normIDF;
            
    return score;
  }

  @Override
  public String getName() {
    return "Default";
  }
}
