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
package edu.cmu.lti.oaqa.knn4qa.utils;

import no.uib.cipr.matrix.DenseVector;

public class VectorUtils {

  
  public static String toString(DenseVector vec) {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < vec.size(); ++i) {
      sb.append(i + ":" + vec.get(i) + " ");
    }
    
    return sb.toString();
  }
  
  public static DenseVector fill(float val, int qty) {
    DenseVector res = new DenseVector(qty);
    for (int i = 0; i < qty; ++i) {
      res.set(i, val);
    }
    return res;
  }
  
}
