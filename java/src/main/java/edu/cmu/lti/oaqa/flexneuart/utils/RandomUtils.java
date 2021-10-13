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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.util.ArrayList;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RandomUtils {
  private static final Logger logger = LoggerFactory.getLogger(RandomUtils.class);

  public RandomUtils(int seed) {
    logger.info(String.format("New random generator with seed: %d", seed));
    mRandGen = new Random(seed);
  }
  
  public <ElemType> ArrayList<ElemType> reservoirSampling(ElemType [] inp, int n) {
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
  
  /***
   * @return returns the next pseudorandom, uniformly distributed float value between 0.0 and 1.0.
   */
  public float nextFloat() {
    return mRandGen.nextFloat() ;
  }
  
  /***
   * @return returns the next pseudorandom, uniformly distributed int value.
   */
  public int nextInt() {
    return mRandGen.nextInt();
  }

  /**
   * Weighted sampling with replacement.
   * 
   * @param weights non-negative weights
   * @param qty number of sampling attempts
   * @return an array of sampled IDs.
   */
  public int[] sampleWeightWithReplace(float weights[], int qty) {
    int res[] = new int[qty];
    
    float totWght = 0;
    for (float e : weights) {
      if (e < 0) { // Let's allow zero weights as long as the sum is positive
        throw new RuntimeException("Weights must be non-negative!");
      }
      totWght += e;
    }
    
    if (qty <= 0) {
      throw new RuntimeException("# of samples must be > 0");
    }
    if (weights.length == 0) {
      throw new RuntimeException("weight array should be non-empty");
    }
    if (totWght <= 0) {
      throw new RuntimeException("Sum of weights needs to be positive!");
    }
    
    for (int att = 0; att < qty; ++att) {
      float sampleWght = mRandGen.nextFloat() * totWght;
      
      float s = 0;
      for (int k = 0; k < weights.length; ++k) {
        float sNext = s + weights[k];
        if (s <= sampleWght && sampleWght < sNext) { // If we have a zero weight an element is never selected
          res[att] = k;
          break;
        }
        s = sNext;
      }
    }
    
    return res;
  }
  
  private final Random mRandGen;
}
