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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.knn4qa.utils.IdValPair;
import edu.cmu.lti.oaqa.knn4qa.utils.IdValParamByValDesc;
import no.uib.cipr.matrix.DenseVector;

/**
 * Re-ranking RM3 similarity (i.e., without extra retrieval step),
 * largely as described in 
 * Condensed List Relevance Models, Fernando Diaz, ICTIR 2015.
 * but with a BM25 scorer instead of the QL.
 * However, the paper might have an error in Eq. (4), b/c
 * it seems that the first term should be one summand
 * from Eq. (2).
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtractorRM3Similarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "RM3Similarity";
  
  public static String TOP_DOC_QTY_PARAM = "topDocQty";
  public static String TOP_TERM_QTY_PARAM = "topTermQty";
  public static String ORIG_WEIGHT_PARAM = "origWeight";

  public FeatExtractorRM3Similarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());

    mSimilObj = new BM25SimilarityLuceneNorm(
                                          conf.getParam(CommonParams.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                                          conf.getParam(CommonParams.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                                          mFieldIndex);
    
    mTopDocQty = conf.getReqParamInt(TOP_DOC_QTY_PARAM);
    mTopTermQty = conf.getReqParamInt(TOP_TERM_QTY_PARAM);
    mOrigWeight = conf.getReqParamFloat(ORIG_WEIGHT_PARAM);
    if (mOrigWeight < 0 || mOrigWeight > 1) {
      throw new Exception(ORIG_WEIGHT_PARAM + " must be >=0 and <=1!");
    }
  }
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData) throws Exception {
    float zNorm = 0;
    int docQty = arrDocIds.size();
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryData);
    HashSet<String> topTerms = new HashSet<String>();
    ArrayList<IdValPair> topDocTerms = new ArrayList<IdValPair>();
    for (int i = 0; i < Math.min(mTopDocQty, docQty); ++i) {
      String docId = arrDocIds.get(i);
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      // 1. Update normalization constant
      zNorm += mSimilObj.compute(queryEntry, docEntry);
      // 2. Extract top terms.
      topDocTerms.clear();
      for (int iDoc = 0; iDoc < docEntry.mWordIds.length; ++iDoc) {
        int wordId = docEntry.mWordIds[iDoc];
        float score = mSimilObj.getDocTermScore(docEntry, iDoc);
        topDocTerms.add(new IdValPair(wordId, score));
      }
      topDocTerms.sort(mScoreSorter);
      for (int k = 0; k < Math.min(mTopTermQty, topDocTerms.size()); ++k) {
        topTerms.add(mFieldIndex.getWord(topDocTerms.get(k).mId));
      }
    }
    /*
     * To prevent division by zero if we have no candidate entries or there're no
     * common terms between queries and documents.
     */
    zNorm = 1.0f / Math.max(zNorm, Float.MIN_NORMAL);
    
    String [] topTermWordArr = {};
    DocEntryParsed queryRM1Entry = mFieldIndex.createDocEntryParsed(topTerms.toArray(topTermWordArr), false);
    
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty());
    
    for (String docId : arrDocIds) {
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      float score = mSimilObj.compute(queryEntry, docEntry) * (
                    mOrigWeight  + (1-mOrigWeight) * zNorm * mSimilObj.compute(queryRM1Entry, docEntry));
      
      v.set(0, score);
    }
    
    return res;
  }

  final BM25SimilarityLuceneNorm   mSimilObj;
  final ForwardIndex               mFieldIndex;
  final int                        mTopDocQty;
  final int                        mTopTermQty;
  final float                      mOrigWeight;
  final IdValParamByValDesc        mScoreSorter = new IdValParamByValDesc();

  @Override
  public int getFeatureQty() {
    return 1;
  }
  
}
