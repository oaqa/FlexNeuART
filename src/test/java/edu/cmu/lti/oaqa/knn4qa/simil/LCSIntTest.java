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

import static org.junit.Assert.*;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.AbstractTest;
import edu.cmu.lti.oaqa.knn4qa.simil_func.DistanceFunctions;


/**
 * @author Leonid Boytsov
 */
public class LCSIntTest extends AbstractTest  {
  final static Logger mLogger = LoggerFactory.getLogger(LCSIntTest.class);
  
  int LCSWrapper(String s1, String s2) {
    String emptyArr[] = new String[0];
    String [] arr1 = s1.isEmpty() ? emptyArr : s1.split(" ");
    String [] arr2 = s2.isEmpty() ? emptyArr : s2.split(" ");
    
    int seq1[] = new int[arr1.length];
    int seq2[] = new int[arr2.length];
    
    for (int i = 0; i < arr1.length; ++i) seq1[i] = Integer.parseInt(arr1[i]);
    for (int i = 0; i < arr2.length; ++i) seq2[i] = Integer.parseInt(arr2[i]);
    
    return DistanceFunctions.compLCS(seq1, seq2);
  }
  
  void LCSWrapper(String s1, String s2, float expValue) {
    float res = LCSWrapper(s1, s2);
    
    assertTrue(approxEqual(mLogger, res, expValue, 1e-8));  
  }
  
  
  @Test
  public void test1() throws Exception {
    LCSWrapper("", "1 2 3 4 5", 0);
    LCSWrapper("1 2 3 4 5", "", 0);
    LCSWrapper("1 2 3 4 5", "2 4 5", 3);
    LCSWrapper("2 4 5", "1 2 3 4 5", 3);
    LCSWrapper("1 2 3 10 4 5", "6 6 6 11 1 2 3 4 5 10 10 10 10", 5);
    LCSWrapper("6 6 6 11 1 2 3 4 5 10 10 10 10", "1 2 3 10 4 5", 5);
    LCSWrapper("1 2 4 3 5", "6 6 6 11 1 2 3 4 5 10 10 10 10", 4);
  }
  
}
