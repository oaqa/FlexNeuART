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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLuceneNorm;
import no.uib.cipr.matrix.DenseVector;

/**
 * A single-field feature extractor that calls an external app to do the hard work.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrSingleFeatExternal extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "External";
  
  public static String BM25_SIMIL = "bm25";
  public static String K1_PARAM = "k1";
  public static String B_PARAM = "b";
  public static String FEAT_APP_PATH = "featAppPath";
  public static String TEMP_FILE_PREFIX = "FeatExtrSingleFeatExternal_tmp";

  public FeatExtrSingleFeatExternal(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    // getReqParamStr throws an exception if the parameter is not defined
    mFieldName = conf.getReqParamStr(FeatExtrConfig.FIELD_NAME);
    String similType = conf.getReqParamStr(FeatExtrConfig.SIMIL_TYPE);
    
    mFeatAppPath = conf.getReqParamStr(FEAT_APP_PATH);

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
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty()); 
    DocEntry queryEntry = getQueryEntry(mFieldName, mFieldIndex, queryData);
    if (queryEntry == null) return res;
    
    File inputFile = File.createTempFile(TEMP_FILE_PREFIX, "input");
    inputFile.deleteOnExit();
    File outputFile = File.createTempFile(TEMP_FILE_PREFIX, "input");
    outputFile.deleteOnExit();
    
    // TODO write 
    
    //JsonCandsInfo candIfno = new JsonCandsInfo();
    
    // Need code in DocEntry to generate candidate info from the query/document entry

    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
    }
    
    
    ProcessBuilder pb =
        new ProcessBuilder(mFeatAppPath, inputFile.getAbsolutePath(), outputFile.getAbsolutePath());
    
    pb.start().wait();
    
    
    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      
      float score = 0;
      // TODO compute the score

      DenseVector v = new DenseVector(1);
      v.set(0, score);      
    }    
    

    
    return res;
  }
  

  @Override
  public String getFieldName() {
    return mFieldName;
  }
  
  final String                       mFieldName;
  final String                       mFeatAppPath;
  final BM25SimilarityLuceneNorm     mSimilObj;
  final ForwardIndex                 mFieldIndex;

  @Override
  public int getDim() {
    // TODO Auto-generated method stub
    return 0;
  }


  @Override
  public String getName() {
    // TODO Auto-generated method stub
    return null;
  }


  @Override
  public int getFeatureQty() {
    // TODO Auto-generated method stub
    return 0;
  }
}
