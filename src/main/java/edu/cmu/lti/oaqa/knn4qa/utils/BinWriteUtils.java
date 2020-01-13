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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.OutputStream;
import java.nio.ByteBuffer;

import edu.cmu.lti.oaqa.knn4qa.simil_func.TrulySparseVector;

/**
 * A few helper functions to write data in the binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class BinWriteUtils {
  
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
    out.write(BinWriteUtils.intToBytes(id.length()));
    out.write(id.getBytes());
  }
  
}
