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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import java.util.ArrayList;

import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.IdValPair;

public class TrulySparseVector {
  

  public final int   mIDs[];
  public final float mVals[];
  
  public TrulySparseVector(int qty) {
    this.mIDs  = new int[qty];
    this.mVals = new float[qty];
  }
  
  public TrulySparseVector(ArrayList<IdValPair> elems) {
    int qty = elems.size();
    this.mIDs  = new int[qty];
    this.mVals = new float[qty];   
    for (int i = 0; i < qty; ++i) {
      mIDs[i] = elems.get(i).mId;
      mVals[i] = elems.get(i).mVal;
    }
  }
  
  public int size() {
    return mIDs.length;
  }
  
  public float l2Norm() {
    float sumSquared = 0;
    
    for (float s : mVals) {
      sumSquared += s * s;
    }
    
    return (float) Math.sqrt(Math.max(Const.FLOAT_EPS, sumSquared));
  }
  
  public void print() {
    for (int i =0 ; i < mIDs.length; ++i) {
      System.out.print(mIDs[i] + ":" + mVals[i] + " ");
    }
    System.out.println("\n==================================");    
  }

}
