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

  protected final String mQueryFieldName;
  protected final String mIndexFieldName;
}
