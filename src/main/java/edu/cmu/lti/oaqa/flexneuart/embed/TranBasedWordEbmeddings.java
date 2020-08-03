/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.flexneuart.embed;

import no.uib.cipr.matrix.Vector.Norm;
import no.uib.cipr.matrix.sparse.SparseVector;

public class TranBasedWordEbmeddings {
  
  public TranBasedWordEbmeddings(SparseVector embedVector, 
                                 SparseVector embedVectorTFxIDF,
                                 SparseVector embedVectorTFProb) {
    mEmbedVector        = embedVector;
    mEmbedVectorTFxIDF  = embedVectorTFxIDF;
    mEmbedVectorTFProb  = embedVectorTFProb;
    mEmbedVectorNorm       = Math.max(mEmbedVector.norm(Norm.Two), Float.MIN_NORMAL);
    mEmbedVectorL1Norm     = Math.max(mEmbedVector.norm(Norm.One), Float.MIN_NORMAL);
    mEmbedVectorTFxIDFNorm = Math.max(mEmbedVectorTFxIDF.norm(Norm.Two), Float.MIN_NORMAL);
    mEmbedVectorTFProbNorm = Math.max(mEmbedVectorTFProb.norm(Norm.Two), Float.MIN_NORMAL);
  }
  public final SparseVector mEmbedVector;  
  public final SparseVector mEmbedVectorTFxIDF;
  public final SparseVector mEmbedVectorTFProb; 
  public final double mEmbedVectorL1Norm;
  public final double mEmbedVectorNorm;  
  public final double mEmbedVectorTFxIDFNorm;
  public final double mEmbedVectorTFProbNorm;
}
