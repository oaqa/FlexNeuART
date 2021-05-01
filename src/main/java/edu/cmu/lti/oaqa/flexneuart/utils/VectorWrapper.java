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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.OutputStream;
import java.io.IOException;

import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;

/**
 * A simple wrapper for a vector that can be either dense or sparse.
 * @author Leonid Boytsov
 *
 */
public class VectorWrapper {
  
  public void write(OutputStream out) throws IOException {

    if (mDenseVec != null) {
      writeDenseVect(mDenseVec, out); 
    }
    if (mSparseVector != null) {
      writeSparseVect(mSparseVector, out);
    }
   
  }
  
  public static void writeSparseVect(TrulySparseVector vec, OutputStream out) throws IOException {
    out.write(BinReadWriteUtils.intToBytes(vec.mIDs.length));
    byte buf[] = BinReadWriteUtils.sparseVectorToBytes(vec);
    out.write(buf);
  }
  
  public static void writeDenseVect(float vec[], OutputStream out) throws IOException {
    out.write(BinReadWriteUtils.intToBytes(vec.length));
    byte buf[] = BinReadWriteUtils.denseVectorToBytes(vec); 
    out.write(buf);
  }
  
  public int qty() {
    return mDenseVec != null ? mDenseVec.length : mSparseVector.size();
  }
  
  public boolean isSparse() {
    return mDenseVec == null;
  }
  
  public VectorWrapper(float [] vec) {
    mDenseVec = vec;
    mSparseVector = null;
  }
  
  public VectorWrapper(TrulySparseVector vec) {
    mSparseVector = vec;
    mDenseVec = null;
  }
  
  public static float scalarProduct(VectorWrapper vec1, VectorWrapper vec2) throws Exception {
    if (vec1.isSparse() != vec2.isSparse()) {
      throw new Exception("Computing scalar product between vectors of incompatible sparsity type!");
    }
    if (vec1.isSparse()) {
      return TrulySparseVector.scalarProduct(vec1.mSparseVector, vec2.mSparseVector);
    } else {
      return DistanceFunctions.compScalar(vec1.mDenseVec, vec2.mDenseVec);
    }
  }

  public int getIdByIdx(int idx) {
    if (mDenseVec != null) {
      return idx;
    } else {
      return mSparseVector.mIDs[idx];
    }
  }
  
  public float getValByIdx(int idx) {
    if (mDenseVec != null) {
      return mDenseVec[idx];
    } else {
      return mSparseVector.mVals[idx];
    }
  }
  
  private final float[] mDenseVec;
  private final TrulySparseVector mSparseVector;

}
