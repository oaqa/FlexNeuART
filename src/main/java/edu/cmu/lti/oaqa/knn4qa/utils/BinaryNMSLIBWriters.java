package edu.cmu.lti.oaqa.knn4qa.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;

/**
 * Various helper function to export data to NMSLIB in the binary format, as well
 * as to submit NMSLIB queries.
 * 
 * @author Leonid Boytsov
 *
 */

public class BinaryNMSLIBWriters {
  static ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN; // we'll do everything on Intel Linux/Mac

  public static void writeDenseVector(float[] vec, ByteBuffer out) {
    out.order(BYTE_ORDER);
    for (float v : vec) {
      out.putFloat(v);
    }
  }
  
  public static void writeSparseVector(TrulySparseVector vec, ByteBuffer out) {
    out.order(BYTE_ORDER);
    for (int i = 0; i < vec.mIDs.length; ++i) {
      out.putInt(vec.mIDs[i]);
      out.putFloat(vec.mVals[i]);
    }
  }
}
