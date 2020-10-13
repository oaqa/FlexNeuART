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
import edu.cmu.lti.oaqa.flexneuart.simil_func.SDMSimilarityAnserini;
import no.uib.cipr.matrix.DenseVector;

/**
 * A sequential dependency similarity feature extractor that wraps up
 * Anserini's port of SDM.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrSDMSimilarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "SDMSimilarity";
  
  public static String WINDOW_PARAM = "window";
  public static String LAMBDA_T_PARAM = "lambdaT";
  public static String LAMBDA_U_PARAM = "lambdaU";
  public static String LAMBDA_O_PARAM = "lambdaO";


  public FeatExtrSDMSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    // getReqParamStr throws an exception if the parameter is not defined
    //String similType = conf.getReqParamStr(FeatExtrConfig.SIMIL_TYPE);

    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    mSimilObjs[0] = new SDMSimilarityAnserini(conf.getParam(LAMBDA_T_PARAM, 0.5f), 
                                          conf.getParam(LAMBDA_O_PARAM, 0.2f), 
                                          conf.getParam(LAMBDA_U_PARAM, 0.3f), 
                                          conf.getParam(WINDOW_PARAM, 8));
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, Map<String, String> queryData) throws Exception {
    return getSimpleFeatures(cands, queryData, mFieldIndex, mSimilObjs);
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }

  final SDMSimilarityAnserini[]   mSimilObjs = new SDMSimilarityAnserini[1];
  final ForwardIndex              mFieldIndex;
}
