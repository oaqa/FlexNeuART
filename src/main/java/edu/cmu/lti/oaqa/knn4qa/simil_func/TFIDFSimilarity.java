package edu.cmu.lti.oaqa.knn4qa.simil_func;

import java.util.HashMap;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.WordEntry;

public abstract class TFIDFSimilarity implements QueryDocSimilarityFunc {
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
  public synchronized Float getIDF(ForwardIndex fieldIndex, int wordId) {
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
