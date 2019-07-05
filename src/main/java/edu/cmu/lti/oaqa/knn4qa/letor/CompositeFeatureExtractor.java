/*
 *  Copyright 2019 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import no.uib.cipr.matrix.DenseVector;


/**
 * A composite feature extractor simply combines features from several 
 * field-specific component-extractors.
 */
public class CompositeFeatureExtractor extends FeatureExtractor {

  public CompositeFeatureExtractor(FeatExtrResourceManager resMngr, String configFile) throws Exception {
    FeatExtrConfig fullConf = FeatExtrConfig.readConfig(configFile);
    ArrayList<SingleFieldFeatExtractor> compList = new ArrayList<SingleFieldFeatExtractor>();
    
    for (OneFeatExtrConf oneExtrConf : fullConf.extractors) {
      SingleFieldFeatExtractor fe = null;
      String extrType = oneExtrConf.type;
      if (extrType.equalsIgnoreCase(FeatExtrTFIDFSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrTFIDFSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrModel1Similarity.EXTR_TYPE)) {
        fe = new FeatExtrModel1Similarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrWordEmbedSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrWordEmbedSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtractorExternalApacheThrift.EXTR_TYPE)) {
        fe = new FeatExtractorExternalApacheThrift(resMngr, oneExtrConf);
      } else 
        // TODO ideally need to inform about the set of supported extractors
        throw new Exception("Unsupported extractor type: " + extrType);
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
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String,DenseVector> res = FeatureExtractor.initResultSet(arrDocIds, getFeatureQty());
    
    int startFeatId = 0;
    for (SingleFieldFeatExtractor featExtr : mCompExtr) {
      Map<String,DenseVector> subRes = featExtr.getFeatures(arrDocIds, queryData);
      int compQty = featExtr.getFeatureQty();
      for (String docId : arrDocIds) {
        DenseVector dst = res.get(docId);
        DenseVector src = subRes.get(docId);
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
