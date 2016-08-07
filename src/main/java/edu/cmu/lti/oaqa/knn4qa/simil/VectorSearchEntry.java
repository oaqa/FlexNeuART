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
package edu.cmu.lti.oaqa.knn4qa.simil;

public class VectorSearchEntry implements Comparable<VectorSearchEntry> {
  /**
   * An identifier, e.g., a document ID or a word. 
   */
  public final String    mID;
  /**
   * The distance (the smaller is better).
   */
  public final float     mDist;  
  
  public VectorSearchEntry(String mWord, float mSimil) {
    this.mID = mWord;
    this.mDist = mSimil;
  }

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "[Identifier: " + mID + ", distance: " + mDist + "]";
  }

  /* (non-Javadoc)
   * The largest value should go first, this is necessary
   * for the priority queue (where the head is the SMALLEST value).
   */
  @Override
  public int compareTo(VectorSearchEntry o) {
    if (mDist > o.mDist) return -1;
    if (mDist < o.mDist) return 1;
    return 0;
  }
  
}
