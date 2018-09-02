package edu.cmu.lti.oaqa.knn4qa.simil;

import java.util.HashMap;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.WordEntry;

public abstract class TFIDFSimilarity {
  public abstract String getName();
  /**
   * Computes the similarity between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  public abstract float compute(DocEntry query, DocEntry doc);

  /**
   * Computes an IDF value. 
   * 
   * <p>If the word isn't found, NULL is returned.
   * Saves the computed value so it's not recomputed in the future.
   * If the word isn't found, NULL is returned.
   * The actual computation is delegated to the child class.
   * </p> 
   * 
   * @param wordId  the word ID
   * @return the IDF value
   */  
  public synchronized Float getIDF(InMemForwardIndex fieldIndex, int wordId) {
    Float res = mIDFCache.get(wordId);
    if (null == res) {
      WordEntry e = fieldIndex.getWordEntry(wordId);
      if (e != null) {
        res = computeIDF(fieldIndex.getDocQty(), e);
        mIDFCache.put(wordId, res);
      }
    }
    return res;    
  }
  
  protected abstract float computeIDF(float docQty, WordEntry e);
  
  private HashMap<Integer, Float> mIDFCache = new HashMap<Integer, Float>();
}
