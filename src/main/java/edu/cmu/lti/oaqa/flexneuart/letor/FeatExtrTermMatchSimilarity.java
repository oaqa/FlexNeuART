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

import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TermMatchSimilarity;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;

/**
 * A normalized (0 to 1) number of query terms appearing in a document.
 * 
 * @author Leonid Boytsov
 *
 */

public class FeatExtrTermMatchSimilarity extends SingleFieldFeatExtractor  {
  public static String EXTR_TYPE = "TermMatchSimilarity";

  public FeatExtrTermMatchSimilarity(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    super(resMngr, conf);
    
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    
    mSimilObjs[0] = new TermMatchSimilarity();
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeaturesMappedIds(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    return getSimpleFeatures(EXTR_TYPE, cands, queryFields, mFieldIndex, mSimilObjs);
  }
  
  TermMatchSimilarity[] mSimilObjs = new TermMatchSimilarity[1];
  final ForwardIndex                 mFieldIndex;

}
