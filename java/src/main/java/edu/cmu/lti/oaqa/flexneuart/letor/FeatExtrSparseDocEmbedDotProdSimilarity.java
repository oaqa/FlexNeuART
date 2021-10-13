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
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;
import edu.cmu.lti.oaqa.flexneuart.utils.BinReadWriteUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * This feature extractor computes the dot (inner) product between pre-computed *SPARSE* vectors for a documents and a query. 
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrSparseDocEmbedDotProdSimilarity extends SingleFieldInnerProdFeatExtractor {
  public static String EXTR_TYPE = "DocSparseEmbedDotProd";
  final Logger logger = LoggerFactory.getLogger(FeatExtrSparseDocEmbedDotProdSimilarity.class);
  
  public FeatExtrSparseDocEmbedDotProdSimilarity(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    super(resMngr, conf);
    
    String indexFieldName = getIndexFieldName();
 
    mFieldIndex = resMngr.getFwdIndex(indexFieldName);
    mNormalize = conf.getParam(FeatExtrWordEmbedSimilarity.USE_L2_NORM, false);
    
    logger.info("Index field name: " + indexFieldName + " normalize embeddings:? " + mNormalize);
  }

  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public int getDim() {
    return 0;
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  public VectorWrapper getFeatInnerProdQueryVector(byte[] query) throws Exception {
    return new VectorWrapper(BinReadWriteUtils.readPackedSparsedVector(query));
  }
  
  public VectorWrapper getFeatInnerProdDocVector(byte[] doc) throws Exception {
    return new VectorWrapper(BinReadWriteUtils.readPackedSparsedVector(doc));
  }

  @Override
  public Map<String, DenseVector> getFeaturesMappedIds(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, getFeatureQty()); 
    
    String queryId = queryFields.mEntryId;  
    if (queryId == null) {
      throw new Exception("Undefined query ID!");
    }
    byte queryEntry[] = queryFields.getBinary(getQueryFieldName());
    if (queryEntry == null) {
      throw new Exception("Empty query data isn't permissible for extractor: " + EXTR_TYPE + " query ID: " + queryId);
    }
    
    TrulySparseVector queryVect = BinReadWriteUtils.readPackedSparsedVector(queryEntry);

    for (CandidateEntry e : cands) {
      TrulySparseVector docVec = BinReadWriteUtils.readPackedSparsedVector(mFieldIndex.getDocEntryBinary(e.mDocId));
      
      DenseVector v = res.get(e.mDocId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", 
                                          e.mDocId));
      }
      float val = mNormalize ?  DistanceFunctions.compNormScalar(queryVect, docVec) :
                                DistanceFunctions.compScalar(queryVect, docVec);
      
      v.set(0, val);
    }
    
    return res;
  } 

  final ForwardIndex        mFieldIndex;
  final boolean             mNormalize;
}
