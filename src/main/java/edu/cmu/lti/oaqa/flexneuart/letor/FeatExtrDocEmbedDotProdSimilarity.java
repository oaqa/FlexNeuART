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

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.utils.BinReadWriteUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * This feature extractor computes the dot (inner) product between a precomputed
 * document and query vector.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrDocEmbedDotProdSimilarity extends SingleFieldInnerProdFeatExtractor {
  public static String EXTR_TYPE = "DocEmbedDotProd";
  
  FeatExtrDocEmbedDotProdSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
 
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    
    String docIds[] = mFieldIndex.getAllDocIds();
    if (docIds.length == 0) {
      throw new Exception("Cannot work with an empty index!");
    }
    // Infer dimensions from the first available vector
    float vec[] = BinReadWriteUtils.readPackedDenseVector(mFieldIndex.getDocEntryBinary(docIds[0]));
    mDim = vec.length;
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
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, getFeatureQty()); 
    byte queryEntry[] = queryFields.getBinary(getQueryFieldName());
    if (queryEntry == null) return res;
    
    float [] queryVect = BinReadWriteUtils.readPackedDenseVector(queryEntry);

    for (CandidateEntry e : cands) {
      float [] docVec = BinReadWriteUtils.readPackedDenseVector(mFieldIndex.getDocEntryBinary(e.mDocId));
      
      DenseVector v = res.get(e.mDocId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", 
                                          e.mDocId));
      }
      v.set(0, DistanceFunctions.compScalar(queryVect, docVec));
    }
    
    return res;
  } 

  final ForwardIndex        mFieldIndex;
  final int                 mDim;
}
