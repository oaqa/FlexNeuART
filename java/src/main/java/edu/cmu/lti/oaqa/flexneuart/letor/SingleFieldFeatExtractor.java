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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Arrays;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.QueryDocSimilarityFunc;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;

/**
 * A feature extractor interface, which computes one or more similarity scores 
 * for a single pair of query and index fields. 
 * Note that the query field name can be different from the index field name: 
 * If the user does not specify the query field name
 * it is assumed to be equal to the index field name.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class SingleFieldFeatExtractor extends FeatureExtractor {
  private static final Logger logger = LoggerFactory.getLogger(SingleFieldFeatExtractor.class);
  
  public SingleFieldFeatExtractor(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    mIndexFieldName = conf.getReqParamStr(CommonParams.INDEX_FIELD_NAME);
    mQueryFieldName = conf.getParam(CommonParams.QUERY_FIELD_NAME, mIndexFieldName);
    mIdMapFieldName = conf.getParam(CommonParams.ID_MAP_FIELD_NAME, "");
    if (mIdMapFieldName.length() == 0) {
      mIdMapper = null;
    } else {
      logger.info("Using field: " + mIdMapFieldName + " to map entry/document IDs");
      mIdMapper = resMngr.getFwdIndex(mIdMapFieldName);
      if (!mIdMapper.isTextRaw()) {
        throw new Exception("Remapping field " + mIdMapFieldName + " should have the raw text type!");
      }
    }
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
  
  protected void warnEmptyQueryField(Logger logger, String extrType, String queryId) {
    logger.warn("Empty field " + getQueryFieldName() + " query ID: " + queryId + " extractor: " + extrType);
  }
  
  /**
   * Obtains features for a set of documents, this function should be <b>thread-safe!</b>. 
   * This function uses entry IDs, which are not necessarily collection entry IDs. This can be used to score passages
   * using, e.g., a document index.
   * 
   * @param     cands        an array of candidate entries.
   * @param     queryFields  a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}. 

   * @return a map docId -> sparse feature vector
   */
  protected abstract Map<String,DenseVector> getFeaturesMappedIds(CandidateEntry[] cands, 
                                                                  DataEntryFields  queryFields) throws Exception;
  
  /**
   * A wrapper function, which obtains features for a set of documents by calling {@link getFeaturesMappedIds}. 
   * This function uses entry IDs, which must be collection entry IDs. If there is no ID mapper defined in the configuration
   * file, the function simply calls {@link getFeaturesMappedIds} with the provided input. Otherwise, it first
   * maps IDs using the remapping index and calls {@link getFeaturesMappedIds} using mapped IDs.
   * 
   * @param     cands        an array of candidate entries.
   * @param     queryFields  a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}. 

   * @return a map docId -> sparse feature vector
   */
  @Override
  public Map<String,DenseVector> getFeatures(CandidateEntry[] cands, 
                                             DataEntryFields  queryFields) throws Exception {
    if (mIdMapper == null) {
      return getFeaturesMappedIds(cands, queryFields);
    } else {
      CandidateEntry[] candsMappedIdTmp = new CandidateEntry[cands.length];
      HashSet<String>  seen = new HashSet<String>();
      
      int mapUniqQty = 0;
      for (int i = 0; i < cands.length; i++) {
        CandidateEntry e = cands[i];
        String origId = e.mDocId;
        String mappedId = mIdMapper.getDocEntryTextRaw(origId);
        if (mappedId == null) {
          throw new Exception("Cannot map id '" + origId + "' using the field: " + mIdMapFieldName);
        }
        if (!seen.contains(mappedId)) {
          seen.add(mappedId);
          candsMappedIdTmp[mapUniqQty++] = new CandidateEntry(mappedId, e.mScore, e.mOrigScore);
        }
      }

      CandidateEntry[] candsMappedId = Arrays.copyOf(candsMappedIdTmp, mapUniqQty);
      
      Map<String, DenseVector> tmpRes = getFeaturesMappedIds(candsMappedId, queryFields);
      HashMap<String, DenseVector> res = new HashMap<String, DenseVector>();

      for (int i = 0; i < cands.length; i++) {
        CandidateEntry e = cands[i];
        String origId = e.mDocId;
        String mappedId = mIdMapper.getDocEntryTextRaw(origId);
        if (mappedId == null) {
          throw new Exception("Cannot map id '" + origId + "' using the field: " + mIdMapFieldName);
        }
        DenseVector feats = tmpRes.get(mappedId);
        if (feats == null) {
          throw new RuntimeException("Bug no features for mapped id '" + mappedId + "' original id '" + origId + "', the remapping field: " + mIdMapFieldName);
        }
        res.put(origId, feats);
      }

      return res;
    }
  }

  /**
   * A helper function that computes one or more simple single-field similarity
   * 
   * @param extrType    extractor type
   * @param cands       candidate records
   * @param queryFields a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}.
   * @param fieldIndex  a field forward index object.
   * @param similObj    an array of objects that computer features (query-document similarity).
   * @return
   * @throws Exception
   */
  protected Map<String, DenseVector> getSimpleFeatures(String extrType,
                                                       CandidateEntry[] cands, 
                                                       DataEntryFields queryFields,
                                                       ForwardIndex fieldIndex, 
                                                       QueryDocSimilarityFunc[] similObj) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, similObj.length); 
    
    String queryId = queryFields.mEntryId;  
    if (queryId == null) {
      throw new Exception("Undefined query ID!");
    }
    
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), fieldIndex, queryFields);
    if (queryEntry == null) {
      warnEmptyQueryField(logger, extrType, queryId);
      return res;
    }
    
    for (CandidateEntry e : cands) {
      DocEntryParsed docEntry = fieldIndex.getDocEntryParsed(e.mDocId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
      }

      
      DenseVector v = res.get(e.mDocId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", e.mDocId));
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
  protected final String mIdMapFieldName;
  protected final ForwardIndex mIdMapper;
}
