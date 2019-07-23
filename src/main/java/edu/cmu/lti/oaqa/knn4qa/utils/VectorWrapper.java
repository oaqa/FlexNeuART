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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.io.IOException;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.knn4qa.simil_func.TrulySparseVector;
import no.uib.cipr.matrix.DenseVector;

/**
 * A simple wrapper for a vector that can be either dense 
 * or sparse with a bit of additional functionality for 
 * writing these vectors to binary stream. 
 * 
 * @author Leonid Boytsov
 *
 */
public class VectorWrapper {
  
  public void write(OutputStream out) throws IOException {

    if (mDenseVec != null) {
      writeDenseVect(mDenseVec, out); 
    }
    if (mSparseVector != null) {
      writeSparseVect(mSparseVector, out);
    }
   
  }
  
  public static void writeSparseVect(TrulySparseVector vec, OutputStream out) throws IOException {
    out.write(BinWriteUtils.intToBytes(vec.mIDs.length));
    byte buf[] = BinWriteUtils.sparseVectorToBytes(vec);
    out.write(buf);
  }
  
  public static void writeDenseVect(float vec[], OutputStream out) throws IOException {
    out.write(BinWriteUtils.intToBytes(vec.length));
    byte buf[] = BinWriteUtils.denseVectorToBytes(vec); 
    out.write(buf);
  }
  
  public int qty() {
    return mDenseVec != null ? mDenseVec.length : mSparseVector.size();
  }
  
  public boolean isSparse() {
    return mDenseVec == null;
  }
  
  public VectorWrapper(float [] vec) {
    mDenseVec = vec;
    mSparseVector = null;
  }
  
  public VectorWrapper(TrulySparseVector vec) {
    mSparseVector = vec;
    mDenseVec = null;
  }
  
  public static float scalarProduct(VectorWrapper vec1, VectorWrapper vec2) throws Exception {
    if (vec1.isSparse() != vec2.isSparse()) {
      throw new Exception("Computing scalar product between vectors of incompatible sparsity type!");
    }
    if (vec1.isSparse()) {
      return TrulySparseVector.scalarProduct(vec1.mSparseVector, vec2.mSparseVector);
    } else {
      return DistanceFunctions.compScalar(vec1.mDenseVec, vec2.mDenseVec);
    }
  }
  
  /**
   * Write all NMSLIB-compatible vectors to a byte stream (an inner-product 
   * between query and document  is supposed to reproduce, at least approximately, the
   * feature values. If query data is null, we generate a data vector by first 
   * reading document information from one or more forward indices.
   * 
   * @param docId           an optional document whose features we need to generate (can be null)
   * @param queryData       an optional query data (can be null) 
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param out             an output stream
   * 
   * @throws Exception
   */
  public static void writeAllVectorsToNMSLIBStream(String docId, Map<String, String> queryData,
                                            ForwardIndex[] compIndices, 
                                            SingleFieldInnerProdFeatExtractor[] compExtractors,
                                            OutputStream out) throws Exception  {
    int featExtrQty = compIndices.length;
    
    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }
    
    for (int i = 0; i < featExtrQty; ++i) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[i];
      VectorWrapper featVect = getFeatInnerProdVector(docId, queryData, compIndices[i], extr);
      
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[i].getName());
      }
      featVect.write(out);
    }
  }

  /**
   * Create an interleaved version of the feature vectors. Each feature vector value is multiplied by a weight.
   * If query data is null, we generate a data vector by first reading document information
   * from one or more forward indices.
   * 
   * @param docId           an optional document whose features we need to generate (can be null)
   * @param queryData       an optional query data (can be null) 
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param compWeights     a vector of feature weights.
   * 
   * @throws Exception
   */
  public static TrulySparseVector createAnInterleavedFeatureVect(String docId, 
                                                    Map<String, String> queryData,
                                                    ForwardIndex[] compIndices, 
                                                    SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                    DenseVector compWeights) throws Exception  {
    int featExtrQty = compIndices.length;

    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }
    if (featExtrQty != compWeights.size()) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of weights (" + compWeights.size() + ")");
    }
    
    ArrayList<IdValPair> allFeatures = new ArrayList<IdValPair>();
    
    for (int compId = 0; compId < featExtrQty; ++compId) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[compId];
      VectorWrapper featVect = getFeatInnerProdVector(docId, queryData, compIndices[compId], extr);
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[compId].getName());
      }
      int qty = featVect.qty();
      
      //System.out.println("Component id: " + compId + " weight: " + compWeights.get(compId));
      for (int idx = 0; idx < qty; ++idx) {
        int newId = compId + featExtrQty * featVect.getIdByIdx(idx);
        float newVal = featVect.getValByIdx(idx) * (float)compWeights.get(compId);
        allFeatures.add(new IdValPair(newId, newVal));
        //System.out.println(featVect.getIdByIdx(idx) + " -> " + newId);
      }
    }
    Collections.sort(allFeatures);
    return new TrulySparseVector(allFeatures);
  }  
  
  public static void writeAllVectorsInterleavedToNMSLIBStream(String docId, 
                                                              Map<String, String> queryData,
                                                              ForwardIndex[] compIndices, 
                                                              SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                              DenseVector compWeights,
                                                              OutputStream out) throws Exception  {
    
    TrulySparseVector featVec = createAnInterleavedFeatureVect(docId, queryData, compIndices, compExtractors, compWeights);
    
    writeSparseVect(featVec, out);
    
  }

  protected static VectorWrapper 
      getFeatInnerProdVector(String docId, 
                                  Map<String, String> queryData,
                                  ForwardIndex compIndx,
                                  SingleFieldInnerProdFeatExtractor extr) throws Exception {
    DocEntry docEntry = null;
    
    boolean isQuery = queryData != null;
    if (isQuery) {
      String queryFieldName = extr.getQueryFieldName();

      String text = queryData.get(queryFieldName);
      if (text == null) {
        throw new Exception("No query information for field: " + queryFieldName);
      }
      docEntry = compIndx.createDocEntry(StringUtils.splitOnWhiteSpace(text), true); // true means including positions
    } else {
      docEntry = compIndx.getDocEntry(docId);
    }
    VectorWrapper featVect = extr.getFeatInnerProdVector(docEntry, isQuery);
    return featVect;
  }
  
  private int getIdByIdx(int idx) {
    if (mDenseVec != null) {
      return idx;
    } else {
      return mSparseVector.mIDs[idx];
    }
  }
  
  private float getValByIdx(int idx) {
    if (mDenseVec != null) {
      return mDenseVec[idx];
    } else {
      return mSparseVector.mVals[idx];
    }
  }
  
  private final float[] mDenseVec;
  private final TrulySparseVector mSparseVector;

}
