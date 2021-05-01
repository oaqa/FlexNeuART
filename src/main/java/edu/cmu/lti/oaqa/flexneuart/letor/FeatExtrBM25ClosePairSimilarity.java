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
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25CloseOrderPairSimilQueryNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25ClosePairSimilarityQueryNormBase;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25CloseUnorderPairSimilQueryNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrBM25ClosePairSimilarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "BM25ClosePairSimilarity";
  
  public static String QUERY_WINDOW_PARAM = "queryWindow";
  public static String DOC_WINDOW_PARAM = "docWindow";

  public FeatExtrBM25ClosePairSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    
    mSimilObjs[0] = new BM25CloseOrderPairSimilQueryNorm(
                            conf.getParam(CommonParams.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                            conf.getParam(CommonParams.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                            conf.getParam(QUERY_WINDOW_PARAM, 4),
                            conf.getParam(DOC_WINDOW_PARAM, 8),
                            mFieldIndex);
    mSimilObjs[1] = new BM25CloseUnorderPairSimilQueryNorm(
                            conf.getParam(CommonParams.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                            conf.getParam(CommonParams.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                            conf.getParam(QUERY_WINDOW_PARAM, 4),
                            conf.getParam(DOC_WINDOW_PARAM, 8),
                            mFieldIndex);
  }

  @Override
  public String getName() {
    return getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    return getSimpleFeatures(cands, queryFields, mFieldIndex, mSimilObjs);
  }

  @Override
  public int getFeatureQty() {
    return 2;
  }

  final BM25ClosePairSimilarityQueryNormBase[]   mSimilObjs = new BM25ClosePairSimilarityQueryNormBase[2];
  final ForwardIndex                             mFieldIndex;
}
