/*
 *  Copyright 2019 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.letor;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;

/**
 * A single-field feature extractor interface (enforcing 
 * implementation of some common functions).
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class SingleFieldFeatExtractor extends FeatureExtractor {

  /**
   * @return true if generates a sparse feature vector.
   */
  public abstract boolean isSparse();
  
  /**
   * Dimensionality of a dense feature vector.
   * 
   * @return number of dimensions.
   */
  public abstract int getDim();
  
  /**
   * 
   * @return the name of the feature field.
   */
  public abstract String getFieldName();
  
  /**
  /**
   * For features that can be computed as an inner product of 
   * document and query vectors, this function produces
   * a corresponding vector from a DocEntry object.
   * 
   * @param e a DocEntry object
   * @param isQuery true for queries and false for documents.
   * 
   * @return a possibly empty array of vector wrapper objects or null
   *         if the inner-product representation is not possible.
   */
  public abstract VectorWrapper getFeatureVectorsForInnerProd(DocEntry e, boolean isQuery);
  
}
