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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import java.util.HashMap;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.WordEntry;

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
  public abstract float compute(DocEntryParsed query, DocEntryParsed doc);

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
  
  /**
   * Extracts a sparse vector corresponding to a query or a document.
   * These vectors are designed so that a dot product of a query and
   * a document vectors is equal to the value of the respective BM25 similarity.
   * </p>
   * 
   * @param e         a query/document entry
   * @param isQuery   true if is a query entry
   * 
   * @return
   */
  public abstract TrulySparseVector getSparseVector(DocEntryParsed e, boolean isQuery);
  
  protected abstract float computeIDF(float docQty, WordEntry e);
  
  private HashMap<Integer, Float> mIDFCache = new HashMap<Integer, Float>();
}
