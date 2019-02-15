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
import java.util.Map;

import java.io.IOException;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldSingleScoreFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.DistanceFunctions;
import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;

public class VectorWrapper {
  
  public void write(OutputStream out) throws IOException {
    byte[] buf = null;
    if (mDenseVec != null) {
      out.write(BinWriteUtils.intToBytes(mDenseVec.length));
      buf = BinWriteUtils.denseVectorToBytes(mDenseVec); 
    }
    if (mSparseVector != null) {
      out.write(BinWriteUtils.intToBytes(mSparseVector.mIDs.length));
      buf = BinWriteUtils.sparseVectorToBytes(mSparseVector);
    }
    if (buf !=null) {
      out.write(buf);
    }
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
  
  public static void writeAllFeatureVectorsToStream(String docId, Map<String, String> queryData,
                                            ForwardIndex[] compIndices, 
                                            SingleFieldSingleScoreFeatExtractor[] compExtractors,
                                            OutputStream out) throws Exception  {
    int featExtrQty = compIndices.length;
    
    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }
    
    for (int i = 0; i < featExtrQty; ++i) {
      SingleFieldSingleScoreFeatExtractor extr = compExtractors[i];
      DocEntry docEntry = null;
      ForwardIndex indx = compIndices[i];
      boolean isQuery = queryData != null;
      if (isQuery) {
        String fieldName = extr.getFieldName();

        String text = queryData.get(fieldName);
        if (text == null) {
          throw new Exception("No query information for field: " + fieldName);
        }
        docEntry = indx.createDocEntry(UtilConst.splitOnWhiteSpace(text), true); // true means including positions
      } else {
        docEntry = indx.getDocEntry(docId);
      }
      VectorWrapper featVect = extr.getFeatureVectorsForInnerProd(docEntry, isQuery);
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[i].getName());
      }
      featVect.write(out);
    }
  }
  
  private final float[] mDenseVec;
  private final TrulySparseVector mSparseVector;
}
