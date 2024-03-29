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
package edu.cmu.lti.oaqa.flexneuart.resources;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrBM25ClosePairSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrDenseDocEmbedDotProdSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrModel1Similarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrPassRetrScore;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrSDMSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrSparseDocEmbedDotProdSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrTFIDFSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrTermMatchSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrWordEmbedSimilarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtractorExternalApacheThrift;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtractorRM3Similarity;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;


/**
 * A composite feature extractor simply combines features from several 
 * field-specific component-extractors.
 */
public class CompositeFeatureExtractor extends FeatureExtractor {
  public static final String EXTRACTOR_TYPE_PARAM = "type";
  public static final String EXTRACTOR_PARAMS_PARAM = "params";

  /** 
   * A composite feature extraction is a protected function: It is supposed to be called only by 
   * a resource manager and not directly.
   * 
   * @param resMngr
   * @param configFile
   * @throws Exception
   */
  protected CompositeFeatureExtractor(ResourceManager resMngr, String configFile) throws Exception {
    RestrictedJsonConfig fullConf = RestrictedJsonConfig.readConfig("composite feature extractor", configFile);
    if (!fullConf.isArray()) {
      throw new Exception("Extractors configuration file: '" + configFile  + "' does not have an array of extractor descriptions!");
    }
    
    ArrayList<SingleFieldFeatExtractor> compList = new ArrayList<SingleFieldFeatExtractor>();
    
    int extrId = 0;
    for (RestrictedJsonConfig oneExtrConfTopLevel : fullConf.getParamConfigArray()) {
      extrId++;
      SingleFieldFeatExtractor fe = null;
      String extrType = oneExtrConfTopLevel.getReqParamStr(EXTRACTOR_TYPE_PARAM);
      RestrictedJsonConfig oneExtrConf = oneExtrConfTopLevel.getParamNestedConfig(EXTRACTOR_PARAMS_PARAM);
      
      if (extrType == null) {
        throw new Exception("Missing value: " + EXTRACTOR_TYPE_PARAM + " for exttractor # " + extrId + " file: " + configFile);
      }
      if (extrType.equalsIgnoreCase(FeatExtrTFIDFSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrTFIDFSimilarity(resMngr, oneExtrConf);
      } else if(extrType.equalsIgnoreCase(FeatExtrTermMatchSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrTermMatchSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrModel1Similarity.EXTR_TYPE)) {
        fe = new FeatExtrModel1Similarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrWordEmbedSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrWordEmbedSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrDenseDocEmbedDotProdSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrDenseDocEmbedDotProdSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrSparseDocEmbedDotProdSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrSparseDocEmbedDotProdSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtractorExternalApacheThrift.EXTR_TYPE)) {
        fe = new FeatExtractorExternalApacheThrift(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrSDMSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrSDMSimilarity(resMngr, oneExtrConf); 
      } else if (extrType.equalsIgnoreCase(FeatExtrBM25ClosePairSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrBM25ClosePairSimilarity(resMngr, oneExtrConf); 
      } else if (extrType.equalsIgnoreCase(FeatExtractorRM3Similarity.EXTR_TYPE)) {
        fe = new FeatExtractorRM3Similarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrPassRetrScore.EXTR_TYPE)) {
        fe = new FeatExtrPassRetrScore(resMngr, oneExtrConf);
      } else {
        // TODO ideally, we need a better factory function that could also inform about the list of available extractors
        throw new Exception("Unsupported extractor type: " + extrType);
      }
      compList.add(fe);
    }
    init(compList);
  }
  
  private void init(ArrayList<SingleFieldFeatExtractor> componentExtractors) {
    int fqty = 0;
    mCompExtr = new SingleFieldFeatExtractor[componentExtractors.size()];
    for (int i = 0; i < mCompExtr.length; ++i) {
      SingleFieldFeatExtractor fe = componentExtractors.get(i);
      mCompExtr[i] = fe;
      fqty += fe.getFeatureQty();
    }
    mFeatureQty = fqty;
  }
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    HashMap<String,DenseVector> res = FeatureExtractor.initResultSet(cands, getFeatureQty());
    
    int startFeatId = 0;
    for (SingleFieldFeatExtractor featExtr : mCompExtr) {
      Map<String,DenseVector> subRes = featExtr.getFeatures(cands, queryFields);
      int compQty = featExtr.getFeatureQty();
      for (CandidateEntry e: cands) {
        DenseVector dst = res.get(e.mDocId);
        DenseVector src = subRes.get(e.mDocId);
        for (int fid = 0; fid < compQty; ++fid) {
          dst.set(startFeatId + fid, src.get(fid));
        }
      }
      startFeatId += compQty;
    }
    return res;
  }

  @Override
  public int getFeatureQty() {
    return mFeatureQty;
  }
  
  public SingleFieldFeatExtractor[] getCompExtr() {
    return mCompExtr;
  }
 
  private int mFeatureQty;
  private SingleFieldFeatExtractor[] mCompExtr;
  
}
