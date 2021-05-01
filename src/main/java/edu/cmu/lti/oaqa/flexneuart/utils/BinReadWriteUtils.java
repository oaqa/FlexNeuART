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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.OutputStream;
import java.nio.ByteBuffer;

import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;

/**
 * Helper functions to read/write data from/to the binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class BinReadWriteUtils {
  final static int BIN_DATA_DENSE_VECTOR = 0;
  final static int BIN_DATA_SPARSE_VECTOR = 1;
  
  /**
   * Converts a 32-bit integer to a sequence of bytes in a given order.
   * 
   * @param v
   * @return
   */
  public static byte[] intToBytes(int v) {
    ByteBuffer out = ByteBuffer.allocate(4);
    out.order(Const.BYTE_ORDER);
    out.putInt(v);
    // The array should be fully filled up
    return out.array();
  }
  
  /**
   * Converts a 64-bit integer to a sequence of bytes.
   * 
   * @param v
   * @return
   */
  public static byte[] longToBytes(long v) {
    ByteBuffer out = ByteBuffer.allocate(8);
    out.order(Const.BYTE_ORDER);
    out.putLong(v);
    // The array should be fully filled up
    return out.array();
  }

  public static byte[] denseVectorToBytes(float[] vec) {
    ByteBuffer out = ByteBuffer.allocate(4 * vec.length);
    out.order(Const.BYTE_ORDER);
    for (float v : vec) {
      out.putFloat(v);
    }
    // The array should be fully filled up
    return out.array();
  }
  
  public static float[] readPackedDenseVector(byte [] data) {
    int qty4 = data.length / 4;
    if (qty4 * 4 < data.length) {
      throw new RuntimeException("Data size: " + data.length + " isn't divisible by 4.");
    }
    if (qty4 <2) {
      throw new RuntimeException("Data size: " + data.length + " too small.");
    }
    ByteBuffer in = ByteBuffer.wrap(data);
    in.order(Const.BYTE_ORDER);
    float [] res = new float[qty4 - 1];
    int type = in.getInt();
    if (type != BIN_DATA_DENSE_VECTOR) {
      throw new RuntimeException("Data type code: " + type + " is not the code for dense vectors.");
    }
    for (int i = 0; i < qty4 - 1; i++) {
      res[i] = in.getFloat();
    }
    return res;
  }
  
  public static byte[] sparseVectorToBytes(TrulySparseVector vec) {
    ByteBuffer out = ByteBuffer.allocate(8 * vec.mIDs.length);
    out.order(Const.BYTE_ORDER);
    for (int i = 0; i < vec.mIDs.length; ++i) {
      out.putInt(vec.mIDs[i]);
      out.putFloat(vec.mVals[i]);
    }
    // The array should be fully filled up
    return out.array();
  }
  
  public static TrulySparseVector readPackedSparsedVector(byte [] data) {
    int qty4 = data.length / 4;
    if (qty4 * 4 < data.length) {
      throw new RuntimeException("Data size: " + data.length + " isn't divisible by 4.");
    }
    if (qty4 < 3) {
      throw new RuntimeException("Data size: " + data.length + " too small.");
    }
    int qty2 = (qty4 - 1) / 2;
    if ((qty4 - 1) > qty2 * 2) {
      throw new RuntimeException("Data size: " + data.length + " isn't appropriate for packed sparse data.");
    }

    ByteBuffer in = ByteBuffer.wrap(data);
    in.order(Const.BYTE_ORDER);
    TrulySparseVector res = new TrulySparseVector(qty2);
    int type = in.getInt();
    if (type != BIN_DATA_SPARSE_VECTOR) {
      throw new RuntimeException("Data type code: " + type + " is not the code for dense vectors.");
    }
    for (int i = 0; i < qty2 - 1; i++) {
      res.mIDs[i] = in.getInt();
      res.mVals[i] = in.getFloat();
    }
    return res;
  }
  
  /**
   * Write a string ID to the stream. To simplify things,
   * we don't permit non-ASCII characters here for the fear
   * that non-ASCII characters will not be preserved correctly
   * by NMSLIB query server.
   * 
   * @param id      input string
   * @param out     output stream
   * @throws Exception
   */
  public static void writeStringId(String id, 
                                   OutputStream out) throws Exception {
    // Here we make a fat assumption that the string doesn't contain any non-ascii characters
    if (StringUtils.hasNonAscii(id)) {
      throw new Exception("Invalid id, contains non-ASCII chars: " + id);
    }
    out.write(BinReadWriteUtils.intToBytes(id.length()));
    out.write(id.getBytes());
  }
  
}
