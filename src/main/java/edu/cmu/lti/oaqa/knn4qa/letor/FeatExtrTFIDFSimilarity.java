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

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A TFxIDF similarity feature extractor that currently supports only BM25.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrTFIDFSimilarity extends SingleFieldSingleScoreFeatExtractor  {
  public static String EXTR_TYPE = "TFIDFSimilarity";
  
  public static String BM25_SIMIL = "bm25";
  public static String K1_PARAM = "k1";
  public static String B_PARAM = "b";
  
  FeatExtrTFIDFSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    // getReqParamStr throws an exception if the parameter is not defined
    mFieldName = conf.getReqParamStr(FeatExtrConfig.FIELD_NAME);
    String similType = conf.getReqParamStr(FeatExtrConfig.SIMIL_TYPE);

    mFieldIndex = resMngr.getFwdIndex(mFieldName);

    if (similType.equalsIgnoreCase(BM25_SIMIL))
      mSimilObj = new BM25SimilarityLuceneNorm(
                                          conf.getParam(K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                                          conf.getParam(B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                                          mFieldIndex);
    else
      throw new Exception("Unsupported field similarity: " + similType);
 
  }
  

  @Override
  public String getFieldName() {
    return mFieldName;
  }
  
  @Override
  public VectorWrapper getFeatInnerProdVector(DocEntry e, boolean isQuery) {
    return new VectorWrapper(mSimilObj.getBM25SparseVector(e, isQuery, true /* share IDF */));
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty()); 
    DocEntry queryEntry = getQueryEntry(mFieldName, mFieldIndex, queryData);
    if (queryEntry == null) return res;
    
    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      float score = mSimilObj.compute(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      v.set(0, score);      
    }    
    
    return res;
  }
  
  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public int getDim() {
    return 0;
  }

  final String                       mFieldName;
  final BM25SimilarityLuceneNorm     mSimilObj;
  final ForwardIndex                 mFieldIndex;

}
