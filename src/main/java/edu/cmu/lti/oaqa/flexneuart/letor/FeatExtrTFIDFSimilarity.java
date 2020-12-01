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
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.CosineTextSimilarity;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A TFxIDF similarity feature extractor that currently supports only BM25.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrTFIDFSimilarity extends SingleFieldInnerProdFeatExtractor  {
  public static String EXTR_TYPE = "TFIDFSimilarity";
  
  FeatExtrTFIDFSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    // getReqParamStr throws an exception if the parameter is not defined
    String similType = conf.getReqParamStr(CommonParams.SIMIL_TYPE);

    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());

    // It's a bit less then ideal to store these names in the CommonParams rather than
    // in a static class variable. However, we have only two of these so far and it's 
    // unlikely we're going have many more in the future (if at all).
    if (similType.equalsIgnoreCase(CommonParams.BM25_SIMIL))
      mSimilObjs[0] = new BM25SimilarityLuceneNorm(
                                          conf.getParam(CommonParams.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                                          conf.getParam(CommonParams.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                                          mFieldIndex);
    if (similType.equalsIgnoreCase(CommonParams.COSINE_SIMIL))
      mSimilObjs[0] = new CosineTextSimilarity(mFieldIndex);
    else
      throw new Exception("Unsupported field similarity: " + similType);
 
  }

  @Override
  public VectorWrapper getFeatInnerProdQueryVector(DocEntryParsed query) throws Exception {
    return new VectorWrapper(mSimilObjs[0].getSparseVector(query, true /* query */));
  }
  
  @Override
  public VectorWrapper getFeatInnerProdDocVector(DocEntryParsed doc) throws Exception {
    return new VectorWrapper(mSimilObjs[0].getSparseVector(doc, false /* doc: not query */));
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
  public boolean isSparse() {
    return true;
  }

  @Override
  public int getDim() {
    return 0;
  }

  final TFIDFSimilarity[]            mSimilObjs = new TFIDFSimilarity[1];
  final ForwardIndex                 mFieldIndex;

}
