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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;

public class BinWriteUtils {
  
  public static ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN; // we'll do everything on Intel Linux/Mac
  
  /**
   * Converts an integer to a sequence of bytes in a given order.
   * 
   * @param v
   * @return
   */
  public static byte[] intToBytes(int v) {
    ByteBuffer out = ByteBuffer.allocate(4);
    out.order(BYTE_ORDER);
    out.putInt(v);
    // The array should be fully filled up
    return out.array();
  }

  public static byte[] denseVectorToBytes(float[] vec) {
    ByteBuffer out = ByteBuffer.allocate(4 * vec.length);
    out.order(BinWriteUtils.BYTE_ORDER);
    for (float v : vec) {
      out.putFloat(v);
    }
    // The array should be fully filled up
    return out.array();
  }
  
  public static byte[] sparseVectorToBytes(TrulySparseVector vec) {
    ByteBuffer out = ByteBuffer.allocate(8 * vec.mIDs.length);
    out.order(BinWriteUtils.BYTE_ORDER);
    for (int i = 0; i < vec.mIDs.length; ++i) {
      out.putInt(vec.mIDs[i]);
      out.putFloat(vec.mVals[i]);
    }
    // The array should be fully filled up
    return out.array();
  }
  
}
