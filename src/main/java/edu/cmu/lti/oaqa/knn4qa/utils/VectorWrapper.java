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
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
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
  
  public VectorWrapper(float [] vec) {
    mDenseVec = vec;
  }
  
  public VectorWrapper(TrulySparseVector vec) {
    mSparseVector = vec;
  }
  
  
  public static void writeAllFeatureVectorsToStream(String docId, Map<String, String> queryData,
                                            ForwardIndex[] compIndices, SingleFieldFeatExtractor[] compExtractors,
                                            OutputStream out) throws Exception  {
    int featExtrQty = compIndices.length;
    
    if (featExtrQty != compExtractors.length) {
      throw new RuntimeException("Bug: number of forward indices (" + featExtrQty + ") != number of extractors (" + compExtractors.length + ")");
    }
    
    for (int i = 0; i < featExtrQty; ++i) {
      SingleFieldFeatExtractor extr = compExtractors[i];
      DocEntry docEntry = null;
      ForwardIndex indx = compIndices[i];
      if (queryData == null) {
        docEntry = indx.getDocEntry(docId);
      } else {
        String fieldName = extr.getFieldName();

        String text = queryData.get(fieldName);
        if (text == null) {
          throw new Exception("No query information for field: " + fieldName);
        }
        docEntry = indx.createDocEntry(UtilConst.splitOnWhiteSpace(text), true); // true means including positions
      }
      VectorWrapper featVect = extr.getFeatureVectorsForInnerProd(docEntry, false);
      if (null == featVect) {
        throw new RuntimeException("Inner product representation is not available for extractor: " + 
                                  compExtractors[i].getName());
      }
      featVect.write(out);
    }
  }
  
  

  
  private float[] mDenseVec;
  private TrulySparseVector mSparseVector;
}
