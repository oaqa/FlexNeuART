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

import java.util.ArrayList;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.AbstractTest;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtils;

public class WMDTest extends AbstractTest {
  final static Logger mLogger = LoggerFactory.getLogger(WMDTest.class);  
  
  float[][] parseDistMatrix(String [] s) {
    float [][]res = new float[s.length][];
    
    for (int i = 0; i < s.length; ++i) {
      String parts[] = StringUtils.splitNoEmpty(s[i], " ");
      res[i] = new float[parts.length];
      for (int k = 0; k < parts.length; ++k)
        res[i][k] = Float.parseFloat(parts[k]);
    }
    
    return res;
  }
  
  float[] getWMD(String dists[], int [] qtys1, int [] qtys2) throws Exception {
    DocEntryParsed doc1 = createFakeDoc(qtys1);
    DocEntryParsed doc2 = createFakeDoc(qtys2);
    
    float [][] distMatr = parseDistMatrix(dists);
    
    return DistanceFunctions.compEMDLike(doc1, doc2, distMatr);
  }
  
  private DocEntryParsed createFakeDoc(int[] qtys) throws Exception {
    ArrayList<Integer> arrQtys = new ArrayList<Integer>(),
                       arrIds  = new ArrayList<Integer>(),
                       wordIdSeq = new ArrayList<Integer>();
    
    for (int q: qtys) {
      arrQtys.add(q);
      arrIds.add(-1); // The test wouldn't care about IDs, only about qtys
    }    
    
    return new DocEntryParsed(arrIds, arrQtys, wordIdSeq, true);
  }
  
  /**
   * This function tests only the first return value and the overall
   * number in the elements in the returned array.
   * 
   * @param dists       A distance matrix encoded as an array of strings.
   * @param qtys1       Number of time the strings repeat in the first document.
   * @param qtys2       Number of time the strings repeat in the second document.
   * @param expValue    An expected number of the first element (the lower-distance bound).
   * @throws Exception 
   */
  void testWrapper(String dists[], int [] qtys1, int [] qtys2, float expValue) throws Exception {
    float [] res = getWMD(dists, qtys1, qtys2);
    assertEquals(res.length, DistanceFunctions.EMD_LIKE_QTY);
    assertTrue(approxEqual(mLogger, res[0], expValue, 1e-6));
  }

  @Test
  public void testOneQtys() throws Exception {
    testWrapper(new String[] {"10 1", "1 10", "10 10"},
                new int[] {1, 1, 1},
                new int[] {1, 1},
                12
               );
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 1, 1},
        new int[] {1, 1, 1},
        3
       );
    testWrapper(new String[] {"10 1 10", "1 10 10", "1 10 10"},
        new int[] {1, 1, 1},
        new int[] {1, 1, 1},
        12
       );    
    
  }
  
  @Test
  public void testReps() throws Exception {
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {2, 1, 1},
        new int[] {1, 1, 1},
        4
       );
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 2, 1},
        new int[] {1, 1, 1},
        4
       );    
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 1, 2},
        new int[] {1, 1, 1},
        4
       );
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 1, 1},
        new int[] {2, 1, 1},        
        4
       );
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 1, 1},
        new int[] {1, 2, 1},        
        4
       );    
    testWrapper(new String[] {"10 1 10", "1 10 10", "10 10 1"},
        new int[] {1, 1, 1},
        new int[] {1, 1, 2},        
        4
       );        
  }
  
  @Test
  public void testInfinity() throws Exception {
    // The third one should always be ignored
    testWrapper(new String[] {"10 1 Infinity", "1 10 Infinity"},
        new int[] {1, 1},
        new int[] {1, 1, 1},
        2
       );
    testWrapper(new String[] {"10 1", "1 10", "Infinity Infinity"},
        new int[] {1, 1, 1},
        new int[] {1, 1},
        2
       );    
    testWrapper(new String[] {"Infinity Infinity", "Infinity Infinity"},
        new int[] {1, 1},
        new int[] {1, 1},
        0
       );    
  }
    
}
