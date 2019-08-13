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

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil_func.QueryDocSimilarityFunc;
import no.uib.cipr.matrix.DenseVector;

/**
 * A single-field feature extractor interface (enforcing 
 * implementation of some common functions). Note that
 * the query-field can be different from an index-field.
 * This permits "between" a single query field such as "text"
 * with multiple document fields, e.g., "title", and "body".
 * If the user does not specify the query field name
 * it is assumed to be equal to the index field name.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class SingleFieldFeatExtractor extends FeatureExtractor {
  
  public SingleFieldFeatExtractor(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    mIndexFieldName = conf.getReqParamStr(FeatExtrConfig.INDEX_FIELD_NAME);
    mQueryFieldName = conf.getParam(FeatExtrConfig.QUERY_FIELD_NAME, mIndexFieldName);
  }

  /**
   * 
   * @return the name of the query field.
   */
  public String getQueryFieldName() {
    return mQueryFieldName;
  }

  /**
   * 
   * @return the name of the index field.
   */
  public String getIndexFieldName() {
    return mIndexFieldName;
  }
  
  /**
   * A helper function that computes a simple single-score similarity
   * 
   * @param arrDocIds       document IDs
   * @param queryData
   * @param fieldIndex
   * @param similObj
   * @return
   * @throws Exception
   */
  protected Map<String, DenseVector> getSimpleFeatures(ArrayList<String> arrDocIds, 
                                                       Map<String, String> queryData,
                                                       ForwardIndex fieldIndex, QueryDocSimilarityFunc[] similObj) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, similObj.length); 
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), fieldIndex, queryData);
    if (queryEntry == null) return res;
    
    for (String docId : arrDocIds) {
      DocEntryParsed docEntry = fieldIndex.getDocEntryParsed(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }

      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      for (int k = 0; k < similObj.length; ++k) {      
        float score = similObj[k].compute(queryEntry, docEntry);
        v.set(k, score);   
      }
    }  
    
    return res;
  }

  protected final String mQueryFieldName;
  protected final String mIndexFieldName;
}
