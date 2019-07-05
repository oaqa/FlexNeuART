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
import edu.cmu.lti.oaqa.knn4qa.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.simil_func.AbstractDistance;
import edu.cmu.lti.oaqa.knn4qa.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtils;

/**
 * @author Leonid Boytsov
 */
public class LCSTest extends AbstractTest {
  final static Logger mLogger = LoggerFactory.getLogger(LCSTest.class);
  
  int LCSWrapper(String s1, String s2) {
    return DistanceFunctions.compLCS(StringUtils.splitNoEmpty(s1, " "), 
                                     StringUtils.splitNoEmpty(s2, " "));
  }
  
  /*
   * It's hard to test the LCS with embeddings, but at least we can verify
   * that for a zero tolerance threshold we get the same result as in the 
   * case of regular LCS.
   */
  void LCSWrapperEmbed(String s1, String s2, EmbeddingReaderAndRecoder wr, float expValue) throws Exception {
    float [][] distMatrix = DistanceFunctions.compDistMatrix(
                                                            AbstractDistance.create("cosine"), 
                                                            StringUtils.splitNoEmpty(s1, " "), 
                                                            StringUtils.splitNoEmpty(s2, " "), 
                                                            wr);
    
    float res[] = DistanceFunctions.compLCSLike(distMatrix, 0.0f);
    
    for (int i = 0; i < 3; ++i) {
      assertTrue(approxEqual(mLogger, res[i], expValue, 1e-8));
    }
    
  }
  
  
  @Test
  public void test1() throws Exception {
    // Each of the words below will have a different vector.
    EmbeddingReaderAndRecoder wr = new EmbeddingReaderAndRecoder("testdata/test_embed/simple.txt", null);
    
    LCSWrapperEmbed("", "a b c d e", wr, 0);
    LCSWrapperEmbed("a b c d e", "", wr, 0);
    LCSWrapperEmbed("a b c d e", "b d e", wr, 3);
    LCSWrapperEmbed("b d e", "a b c d e", wr, 3);
    LCSWrapperEmbed("a b c x d e", "6 6 6 u a b c d e x x x x", wr, 5);
    LCSWrapperEmbed("6 6 6 u a b c d e x x x x", "a b c x d e", wr, 5);
    LCSWrapperEmbed("a b d c e", "6 6 6 u a b c d e x x x x", wr, 4);
  }
  
}
