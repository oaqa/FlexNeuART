/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.giza;

/**
 * A helper class to extract translation entries from GIZA++
 * output files.
 * 
 * @author Leonid Boytsov
 *
 */
public class GizaTranRec implements Comparable<GizaTranRec> {
  public final int     mSrcId;
  public final int     mDstId;
  public final float   mProb;

  GizaTranRec(int srcId, int dstId, float prob) {
    mSrcId = srcId;
    mDstId = dstId;
    mProb  = prob;
  }
  
  public GizaTranRec(String line) throws Exception {
    String parts[] = line.trim().split("\\s+");
    if (parts.length != 3) {
      throw new Exception(
          String.format("Wrong format of line '%s', got %d fields instead of three.",
                        line, parts.length));
    }
    try {
      mSrcId  = Integer.parseInt(parts[0]);
      mDstId  = Integer.parseInt(parts[1]);
      mProb   = Float.parseFloat(parts[2]);
    } catch (NumberFormatException e) {
      throw new Exception(
          String.format("Wrong format of line '%s', either IDs or the probability is not valid.",
                        line));      
    }
    
  }

  @Override
  public int compareTo(GizaTranRec o) {
    if (mSrcId != o.mSrcId) return mSrcId - o.mSrcId;
    return mDstId - o.mDstId;
  }
}
