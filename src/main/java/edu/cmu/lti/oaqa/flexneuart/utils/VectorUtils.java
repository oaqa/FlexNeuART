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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;
import no.uib.cipr.matrix.DenseVector;

/**
 * Vector utility functions including functionality for writing {@link VectorWrapper} vectors to binary stream. 
 * The writing functions implement a lot of tiny details about differences in:
 * 1. generating query and document vectors
 * 2. generating interleaved vectors
 * 3. processing documents in batches
 * 
 * 
 * @author Leonid Boytsov
 *
 */
public class VectorUtils {

  public static String toString(DenseVector vec) {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < vec.size(); ++i) {
      sb.append(i + ":" + vec.get(i) + " ");
    }
    
    return sb.toString();
  }
  
  public static DenseVector fill(float val, int qty) {
    DenseVector res = new DenseVector(qty);
    for (int i = 0; i < qty; ++i) {
      res.set(i, val);
    }
    return res;
  }
  
  
  protected static VectorWrapper getFeatInnerProdQueryVector(DataEntryFields queryFields,
                                                            ForwardIndex compIndx, 
                                                            SingleFieldInnerProdFeatExtractor extr) throws Exception {
      String queryFieldName = extr.getQueryFieldName();

      if (!compIndx.isBinary()) {
        String queryText = queryFields.getString(queryFieldName);
        if (queryText == null) {
          throw new Exception("No query information for field: " + queryFieldName);
        }
        if (compIndx.isTextRaw()) {
          return extr.getFeatInnerProdQueryVector(queryText);
        } else {
          DocEntryParsed queryEntry = compIndx.createDocEntryParsed(StringUtils.splitOnWhiteSpace(queryText), 
                                                                    true); // true means including positions
          return extr.getFeatInnerProdQueryVector(queryEntry);
        }
      } else {
        byte [] queryBinary = queryFields.getBinary(queryFieldName);
        if (queryBinary == null) {
          throw new Exception("No query information for field: " + queryFieldName);
        }
        return extr.getFeatInnerProdQueryVector(queryBinary);
      }
  }
  

  
  /**
   * Create an interleaved sparse vectors by performing a weighted combination
   * of provided vectors.
   * 
   * @param vects        vectors to combine
   * @param weights      vector weights
   * @return
   */
  public static TrulySparseVector interleaveVectors(VectorWrapper vects[], DenseVector vectWeights) {
    int vectQty = vects.length;
    
    if (vectQty != vectWeights.size()) {
      throw new RuntimeException("Bug: number of vectors (" + vectQty + ") != number of weights (" + vectWeights.size() + ")");
    }
    
    ArrayList<IdValPair> allFeatures = new ArrayList<IdValPair>();

    for (int vectId = 0; vectId < vectQty; ++vectId) {
      VectorWrapper v = vects[vectId];
      int elemQty = v.qty();

      for (int idx = 0; idx < elemQty; ++idx) {
        // A new ID is a unique vector offset plus the old ID times the number of vectors.
        // This placement guarantees that new IDs are unique as long the original ones are unique:
        // e.g., if all vectors have a non-zero element with ID K, we will have non-zero elements with IDs:
        // 0 + K * vectQty, 1 + K * vectQty, 2 + K * vectQty, ... < vectQty - 1 + K *vectQty
        int newId = vectId + vectQty * v.getIdByIdx(idx);
        float newVal = v.getValByIdx(idx) * (float) vectWeights.get(vectId);
        allFeatures.add(new IdValPair(newId, newVal));
      }
    }
    Collections.sort(allFeatures);
    return  new TrulySparseVector(allFeatures);
  }

  
  /**
   * Write NMSLIB-compatible inner-product generating query vectors to a byte stream.
   * 
   * @param queryFields     a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}.
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param out             an output stream
   * 
   * @throws Exception
   */
  public static void writeInnerProdQueryVecsToNMSLIBStream(DataEntryFields queryFields,
                                                          ForwardIndex[] compIndices, 
                                                          SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                          OutputStream out) throws Exception  {
    int featExtrQty = compIndices.length;
    
    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }
    
    for (int i = 0; i < featExtrQty; ++i) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[i];
      VectorWrapper featVect = getFeatInnerProdQueryVector(queryFields, compIndices[i], extr);
      
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[i].getName());
      }
      featVect.write(out);
    }
  }
  

  /**
   * Create an interleaved version of the inner-product reproducing query feature vectors. 
   * Each feature vector value is multiplied by a weight.
   * 
   * @param queryFields     a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}.
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param compWeights     a vector of feature weights.
   * 
   * @throws Exception
   */
  public static TrulySparseVector createInterleavedInnerProdQueryFeatVect(
                                                    DataEntryFields queryFields,
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

    VectorWrapper[]   compVects = new VectorWrapper[featExtrQty];
    
    for (int compId = 0; compId < featExtrQty; ++compId) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[compId];
      VectorWrapper featVect = getFeatInnerProdQueryVector(queryFields, compIndices[compId], extr);
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[compId].getName());
      }
      compVects[compId] = featVect;
    }
    
    return interleaveVectors(compVects, compWeights);
  } 
  
  /**
   * Write NMSLIB-compatible inner-product generating query vector to a byte stream
   * by first interleaving and weighting individual sub-vectors.
   * 
   * @param queryFields     a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}.
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param out             an output stream
   * 
   * @throws Exception
   */
  public static void writeInterleavedInnerProdQueryVectToNMSLIBStream(DataEntryFields queryFields,
                                                                     ForwardIndex[] compIndices, 
                                                                     SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                                     DenseVector compWeights,
                                                                     OutputStream out) throws Exception  {
    
    TrulySparseVector featVec = createInterleavedInnerProdQueryFeatVect(queryFields, compIndices, compExtractors, compWeights);
    
    VectorWrapper.writeSparseVect(featVec, out);
  }
  

  /**
   * Write NMSLIB-compatible inner-product generating document vectors to a byte stream.
   * 
   * @param docIds          document IDs
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param out             an output stream
   * 
   * @throws Exception
   */
  public static void writeInnerProdDocVecsBatchToNMSLIBStream(String docIds[],
                                                              ForwardIndex[] compIndices, 
                                                              SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                              OutputStream out) throws Exception  {
    int featExtrQty = compIndices.length;
    
    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }

    ArrayList<VectorWrapper[]>  docVectBatches = new ArrayList<VectorWrapper[]>();
    
    for (int compId = 0; compId < featExtrQty; ++compId) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[compId];
      docVectBatches.add(extr.getFeatInnerProdDocVectorBatch(compIndices[compId], docIds));
    }
    
    for (int resId = 0; resId < docIds.length; ++resId) {
      BinReadWriteUtils.writeStringId(docIds[resId], out);      
      for (int compId = 0; compId < featExtrQty; ++compId) {
        VectorWrapper featVect = docVectBatches.get(compId)[resId];
        if (null == featVect) {
          throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                    compExtractors[compId].getName());
        }
        featVect.write(out);
      }
    }
  }
  
  /**
   * Write NMSLIB-compatible inner-product generating query vector to a byte stream
   * by first interleaving and weighting individual sub-vectors.
   * 
   * @param docIds
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param out             an output stream
   * 
   * @throws Exception
   */
  public static void writeInterleavedInnerProdDoctVectBatchToNMSLIBStream(String docIds[],
                                                                          ForwardIndex[] compIndices, 
                                                                          SingleFieldInnerProdFeatExtractor[] compExtractors,
                                                                          DenseVector compWeights,
                                                                          OutputStream out) throws Exception  {

    TrulySparseVector[] batchVecs = createInterleavedInnerProdDocFeatureVecBatch(docIds, 
                                                                                                                                                          compIndices, 
                                                                                                                                                          compExtractors, 
                                                                                                                                                          compWeights);
    for (int resId = 0; resId < batchVecs.length; resId++) {
      BinReadWriteUtils.writeStringId(docIds[resId], out);  
      VectorWrapper.writeSparseVect(batchVecs[resId], out);
    }
  }

  /**
   * Create an interleaved version of the inner-product reproducing document feature vectors. 
   * Each feature vector value is multiplied by a weight.
   * 
   * @param docIds          an array of document IDs.
   * @param compIndices     forward indices (one per each component extractor)
   * @param compExtractors  component extractor 
   * @param compWeights     a vector of feature weights.
   * 
   * @return an array of {@link TrulySparseVector}
   * 
   * @throws Exception
   */
  public static TrulySparseVector[] createInterleavedInnerProdDocFeatureVecBatch(
                                                    String docIds[], 
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
    
    int resQty = docIds.length;
    TrulySparseVector  res[] = new TrulySparseVector[resQty];
    ArrayList<VectorWrapper[]>  docVectBatches = new ArrayList<VectorWrapper[]>();
    
    for (int compId = 0; compId < featExtrQty; ++compId) {
      SingleFieldInnerProdFeatExtractor extr = compExtractors[compId];
      docVectBatches.add(extr.getFeatInnerProdDocVectorBatch(compIndices[compId], docIds));
    }
    
    VectorWrapper[]   compVects = new VectorWrapper[featExtrQty];
    
    for (int resId = 0; resId < resQty; resId++) {
      for (int compId = 0; compId < featExtrQty; ++compId) {
        VectorWrapper featVect = docVectBatches.get(compId)[resId];
        if (null == featVect) {
          throw new RuntimeException(
              "Inner product representation is not available for extractor: " + compExtractors[compId].getName());
        }
        compVects[compId] = featVect;
      }

      res[resId] = interleaveVectors(compVects, compWeights);
    }
    
    return res;
  }

  
}
