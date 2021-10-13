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
import java.util.Iterator;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.utils.BinReadWriteUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * This feature extractor computes the dot (inner) product between pre-computed *DENSE* vectors for a documents and a query. 
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrDenseDocEmbedDotProdSimilarity extends SingleFieldInnerProdFeatExtractor {
  public static String EXTR_TYPE = "DocDenseEmbedDotProd";
  final Logger logger = LoggerFactory.getLogger(FeatExtrDenseDocEmbedDotProdSimilarity.class);
  
  public FeatExtrDenseDocEmbedDotProdSimilarity(ResourceManager resMngr, RestrictedJsonConfig conf) throws Exception {
    super(resMngr, conf);
    
    String indexFieldName = getIndexFieldName();
 
    mFieldIndex = resMngr.getFwdIndex(indexFieldName);
    
    Iterator<String> docIdIter = mFieldIndex.getDocIdIterator();
    if (!docIdIter.hasNext()) {
      throw new Exception("Cannot work with an empty index!");
    }
    // Infer dimensions from the first available vector
    float vec[] = BinReadWriteUtils.readPackedDenseVector(mFieldIndex.getDocEntryBinary(docIdIter.next()));
    mDim = vec.length;
    
    mNormalize = conf.getParam(FeatExtrWordEmbedSimilarity.USE_L2_NORM, false);
    
    logger.info("Index field name: " + indexFieldName + " normalize embeddings:? " + mNormalize);
  }

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public int getDim() {
    return mDim;
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  public VectorWrapper getFeatInnerProdQueryVector(byte[] query) throws Exception {
    return new VectorWrapper(BinReadWriteUtils.readPackedDenseVector(query));
  }
  
  public VectorWrapper getFeatInnerProdDocVector(byte[] doc) throws Exception {
    return new VectorWrapper(BinReadWriteUtils.readPackedDenseVector(doc));
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
    
    float [] queryVect = BinReadWriteUtils.readPackedDenseVector(queryEntry);

    for (CandidateEntry e : cands) {
      float [] docVec = BinReadWriteUtils.readPackedDenseVector(mFieldIndex.getDocEntryBinary(e.mDocId));
      
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
  final int                 mDim;
  final boolean             mNormalize;
}
