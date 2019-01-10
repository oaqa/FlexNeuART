package edu.cmu.lti.oaqa.knn4qa.utils;

import java.nio.ByteBuffer;

import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;

public class VectorWrapper {
  public void write(ByteBuffer out) {
    if (mDenseVec != null) {
      BinaryNMSLIBWriters.writeDenseVector(mDenseVec, out);
    }
    if (mSparseVector != null) {
      BinaryNMSLIBWriters.writeSparseVector(mSparseVector, out);
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
