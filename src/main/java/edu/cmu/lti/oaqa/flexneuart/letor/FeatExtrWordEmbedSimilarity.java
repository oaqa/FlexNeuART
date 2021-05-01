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
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrWordEmbedSimilarity extends SingleFieldInnerProdFeatExtractor {
  public static String EXTR_TYPE = "avgWordEmbed";
  
  public static String QUERY_EMBED_FILE = "queryEmbedFile";
  public static String DOC_EMBED_FILE = "docEmbedFile";
  public static String USE_TFIDF_WEIGHT = "useIDFWeight";
  public static String USE_L2_NORM = "useL2Norm";
  
  FeatExtrWordEmbedSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
 
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    mSimilObj = new BM25SimilarityLuceneNorm(BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                             BM25SimilarityLucene.DEFAULT_BM25_B, 
                                             mFieldIndex);
    
    mUseIDFWeight = conf.getReqParamBool(USE_TFIDF_WEIGHT);
    mUseL2Norm = conf.getReqParamBool(USE_L2_NORM);
    
    String docEmbedFile = conf.getReqParamStr(DOC_EMBED_FILE);
    mDocEmbed = resMngr.getWordEmbed(getIndexFieldName(), docEmbedFile);
    String queryEmbedFile = conf.getParam(QUERY_EMBED_FILE, null);
    if (queryEmbedFile == null) {
      mQueryEmbed = mDocEmbed;
    } else {
      mQueryEmbed = resMngr.getWordEmbed(getQueryFieldName(), queryEmbedFile);
      if (mQueryEmbed.getDim() != mDocEmbed.getDim()) {
        throw new 
          Exception("Dimension mismatch btween query and document embeddings for field: " + getQueryFieldName());
      }
    }
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  @Override
  public VectorWrapper getFeatInnerProdQueryVector(DocEntryParsed e) throws Exception {
    // note we use query embeddings here
    return new VectorWrapper(mQueryEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                            mUseIDFWeight, 
                            true /* normalize vectors!!!*/ ));
  }
  
  @Override
  public VectorWrapper getFeatInnerProdDocVector(DocEntryParsed e) throws Exception {
    // note that we use document embeddings here
    return new VectorWrapper(mDocEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                            mUseIDFWeight, 
                            true /* normalize vectors!!!*/ ));
  }  

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public int getDim() {
    return mDocEmbed.getDim();
  }
    
  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields) throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, getFeatureQty()); 
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryFields);
    if (queryEntry == null) return res;
    
    float [] queryVect = mQueryEmbed.getDocAverage(queryEntry, mSimilObj, mFieldIndex, 
                                                   mUseIDFWeight, mUseL2Norm);

    for (CandidateEntry e : cands) {
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(e.mDocId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
      }
      
      float [] docVec = mDocEmbed.getDocAverage(docEntry, mSimilObj, mFieldIndex, 
                                                   mUseIDFWeight, mUseL2Norm);

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
  final TFIDFSimilarity     mSimilObj;
  final boolean             mUseIDFWeight;
  final boolean             mUseL2Norm;
  final EmbeddingReaderAndRecoder mDocEmbed;
  final EmbeddingReaderAndRecoder mQueryEmbed;


}
