/*
 *  Copyright 2019+ Carnegie Mellon University
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


import static org.junit.Assert.assertTrue;

import java.util.ArrayList;

import org.junit.Test;

public class RandomUtilsTest {

  private static final boolean DEBUG_PRINT = false;

  void doSimpleTest( int iterQty, int maxNum, int sampleQty, float sigmaQty, int seed) {
    float [] qtys = new float[maxNum + 1];
    Integer [] allInts = new Integer[maxNum + 1];
    RandomUtils rand = new RandomUtils(seed);
    
    for (int k = 0; k <= maxNum; ++k) {
      allInts[k] = k;
    }
    
    
    for (int iter = 0; iter < iterQty; ++iter) {
      ArrayList<Integer> res = rand.reservoirSampling(allInts, sampleQty);
      for (int v : res) {
        qtys[v]++;
      }
    }
    for (int k = 0; k <= maxNum; ++k) {
      qtys[k] /= iterQty;
    }
    
    float p = (float)sampleQty / (maxNum + 1);
    float sigma = (float) (Math.sqrt(p*(1-p)) / Math.sqrt(iterQty));
    
    float dev = 0;
    for (int i = 0 ; i < maxNum; ++i) {
      dev = Math.max(dev, Math.abs(qtys[i] - p));
    }
    if (DEBUG_PRINT) {
      System.out.println("### expAvg=" + p + " expSigma: " + sigma + " dev: " + dev);
      for (int i = 0; i < maxNum; ++i) {
        System.out.print(qtys[i] + " ");
      }
      System.out.println();
    }
    assertTrue("Deviation violation, sigma: " + sigma + " dev:" + dev + " sigmaQty: " + sigmaQty, dev < sigmaQty * sigma);
  }
  
  @Test
  public void test() {
    for (int seed = 0; seed < 5; seed++) {
      doSimpleTest(100 /* iterQty */, 10, 2, 6f /* six-sigma should not be exceeded, very unlikely */, seed);
      doSimpleTest(50 /* iterQty */, 20, 5, 6f /* six-sigma should not be exceeded, very unlikely */, seed);
    }
  }

}
