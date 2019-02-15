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

import java.util.ArrayList;
import java.util.Random;


public class RandomUtils {
  public static <ElemType> ArrayList<ElemType> reservoirSampling(ElemType [] inp, int n) {
    ArrayList<ElemType> res = new ArrayList<ElemType>();
    
    for (int i = 0; i < Math.min(n, inp.length); ++i) {
      res.add(inp[i]);
    }
    for (int i = n; i < inp.length; ++i) {
      int replId = mRandGen.nextInt(i + 1);
      if (replId < n) {
        res.set(replId,inp[i]);
      }
    }
    
    return res;
  }
  
  private final static Random mRandGen = new Random(0);
}
