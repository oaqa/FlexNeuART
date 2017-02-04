/*
 *  Copyright 2016 Carnegie Mellon University
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

public class TrulySparseVector {
  

  public final int   mIDs[];
  public final float mVals[];
  
  public TrulySparseVector(int qty) {
    this.mIDs  = new int[qty];
    this.mVals = new float[qty];
  }
  
  public static float scalarProduct(TrulySparseVector v1, TrulySparseVector v2) {
    float res = 0;
    
    int qty1 = v1.mIDs.length;
    int qty2 = v2.mIDs.length;
    
    int i1 = 0, i2 = 0;
    
    while(i1 < qty1 && i2 < qty2) {
      int wordId1 = v1.mIDs[i1];
      int wordId2 = v2.mIDs[i2];
      if (wordId1 < wordId2) {
        i1++;
      } else if (wordId1 > wordId2) {
        i2++;
      } else {
     /*
      *  Ignore OOV words  (id < 0), if they slip through the cracks.
      *  Note that if wordId1 >= 0, then wordId2 >= 0 too (because word IDs are equal at this point)       
      */
        if (wordId1 >=0) { 
          res += v1.mVals[i1] * v2.mVals[i2];
        }
        i1++; i2++;
      }
    }
    
    return res;
  }
  
  public void print() {
    for (int i =0 ; i < mIDs.length; ++i) {
      System.out.print(mIDs[i] + ":" + mVals[i] + " ");
    }
    System.out.println("\n==================================");    
  }

}
