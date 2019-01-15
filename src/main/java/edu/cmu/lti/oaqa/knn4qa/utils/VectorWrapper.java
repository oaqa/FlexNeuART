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

import java.io.OutputStream;
import java.io.IOException;

import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;

public class VectorWrapper {
  
  public void write(OutputStream out) throws IOException {
    byte[] buf = null;
    if (mDenseVec != null) {
      buf = BinWriteUtils.denseVectorToBytes(mDenseVec); 
    }
    if (mSparseVector != null) {
      buf = BinWriteUtils.sparseVectorToBytes(mSparseVector);
    }
    if (buf !=null) {
      out.write(buf);
    }
  }
  
  public VectorWrapper(float [] vec) {
    mDenseVec = vec;
  }
  
  public VectorWrapper(TrulySparseVector vec) {
    mSparseVector = vec;
  }

  
  private float[] mDenseVec;
  private TrulySparseVector mSparseVector;
}
