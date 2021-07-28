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
package edu.cmu.lti.oaqa.flexneuart.simil;

import static org.junit.Assert.*;

import org.junit.Test;

import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;

public class SparseCosineTest {
  @Test
  public void test1() throws Exception {
    // a quick stupid test of the norm
    TrulySparseVector tmp1 = new TrulySparseVector(10);
    for (int i = 0; i < tmp1.mIDs.length; i++) {
      tmp1.mVals[i] = i + 1;
    }
    TrulySparseVector tmp2 = new TrulySparseVector(20);
    for (int i = 0; i < tmp2.mIDs.length; i++) {
      tmp2.mVals[i] = i + 1;
    }
   
    assertEquals(DistanceFunctions.compScalar(tmp1, tmp2) / (tmp1.l2Norm() * tmp2.l2Norm()),
                 DistanceFunctions.compNormScalar(tmp1, tmp2), 1e-4);
    
    float origNorm = tmp1.l2Norm();
    
    assertEquals(origNorm * origNorm, DistanceFunctions.compScalar(tmp1, tmp1), 1e-4);
    assertEquals(1, DistanceFunctions.compNormScalar(tmp1, tmp1), 1e-4);
    
    for (int i = 0; i < tmp1.mIDs.length; i++) {
      tmp1.mVals[i] /= origNorm;
    }
    
    assertEquals(1, DistanceFunctions.compScalar(tmp1, tmp1), 1e-4);
    assertEquals(1, DistanceFunctions.compNormScalar(tmp1, tmp1), 1e-4);
    
    tmp1 = new TrulySparseVector(2);
    tmp2 = new TrulySparseVector(2);
    
    for (int k = 0; k < tmp1.mVals.length; ++k)
      tmp1.mVals[k] = 1;
    for (int k = 0; k < tmp2.mVals.length; ++k)
      tmp2.mVals[k] = 1;
    
    tmp1.mIDs[0] = 0; tmp1.mIDs[1] = 1;
    tmp2.mIDs[0] = 2; tmp2.mIDs[1] = 3;
    
    assertEquals(0, DistanceFunctions.compScalar(tmp1, tmp2), 1e-4);
    assertEquals(0, DistanceFunctions.compNormScalar(tmp1, tmp2), 1e-4);
    
    tmp1.mIDs[0] = 0; tmp1.mIDs[1] = 1;
    tmp2.mIDs[0] = 1; tmp2.mIDs[1] = 2;
    
    assertEquals(1, DistanceFunctions.compScalar(tmp1, tmp2), 1e-4);
    assertEquals(0.5, DistanceFunctions.compNormScalar(tmp1, tmp2), 1e-4);
  }
}
