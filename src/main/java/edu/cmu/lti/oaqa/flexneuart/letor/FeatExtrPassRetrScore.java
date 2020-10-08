/*
 *  Copyright 2014+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.flexneuart.letor;

import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import no.uib.cipr.matrix.DenseVector;

/**
 * A feature extractor that simply replicates a score from a retriever or
 * an intermediate re-ranker. Although it does not belong to any specific
 * field, for simplicity of refactoring (to avoid breaking things, all
 * other extractors so far has inherited from SingleFieldFeatExtractor)
 * it was inherited from this common extractor type. 
 * Thus, a user has to specify an arbitrary index field 
 * name. However, it is not going to be used.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrPassRetrScore extends SingleFieldFeatExtractor  {
  public static String EXTR_TYPE = "PassRetrScore";
  
  // Always use the original retrieval score even if there's an intermediate re-ranker
  public static String USE_ORIG_RETR_SCORE = "useOrigRetrScore";
  
  FeatExtrPassRetrScore(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    
    mUseOrigScore = conf.getParamBool(USE_ORIG_RETR_SCORE);
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, Map<String, String> queryData) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, 1); 
 
    for (CandidateEntry e : cands) {

      DenseVector v = res.get(e.mDocId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", e.mDocId));
      }
      
      float score = mUseOrigScore ? e.mOrigScore : e.mScore;
      
      v.set(0, score);   
    }
    
    return res;
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }
  
  final boolean mUseOrigScore;
}
