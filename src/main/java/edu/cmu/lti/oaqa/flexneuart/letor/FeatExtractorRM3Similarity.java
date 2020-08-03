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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.flexneuart.utils.IdValPair;
import edu.cmu.lti.oaqa.flexneuart.utils.IdValParamByValDesc;
import no.uib.cipr.matrix.DenseVector;

/**
 * Re-ranking RM3 similarity (i.e., without extra retrieval step),
 * largely as described in 
 * Condensed List Relevance Models, Fernando Diaz, ICTIR 2015.
 * but with a BM25 scorer instead of the QL.
 * That is we replace  p(w|D) with BM25 scores. 
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtractorRM3Similarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "RM3Similarity";
  
  public static String TOP_DOC_QTY_PARAM = "topDocQty";
  public static String TOP_TERM_QTY_PARAM = "topTermQty";
  public static String ORIG_WEIGHT_PARAM = "origWeight";
  
  public static Boolean DEBUG_PRINT = false;

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
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, Map<String, String> queryData) throws Exception {
    
    int docQty = cands.length;
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryData);
    HashMap<Integer, Float>     topTerms = new HashMap<Integer, Float>();
    ArrayList<IdValPair>        topDocTerms = new ArrayList<IdValPair>();
    ArrayList<DocEntryParsed>   queryDocEntries = new ArrayList<DocEntryParsed>();
    ArrayList<Float>            topDocScore = new ArrayList<Float>();
    
    if (DEBUG_PRINT) {
      System.out.println("==========");
      for (int qid : queryEntry.mWordIds) {
        if (qid >= 0) {
          System.out.println(mFieldIndex.getWord(qid));
        }
      }
    }
    
    int topQty = Math.min(mTopDocQty, docQty);
    
    float topDocScoreNorm = 0;
    for (int i = 0; i < docQty; ++i) {
      String docId = cands[i].mDocId;
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      queryDocEntries.add(docEntry);
      
      if (i < topQty) {
        // Compute and memorize a document score with respect to the query
        float score = mSimilObj.compute(queryEntry, docEntry);
        topDocScore.add(score);
        // Update the normalization constant
        topDocScoreNorm += score;
      }
    }
    float invTopDocScoreNorm = (float) (1.0/Math.max(topDocScoreNorm, 1e-9));
    if (DEBUG_PRINT) 
      System.out.println(String.format("!!!! %f %f", topDocScoreNorm, invTopDocScoreNorm));
    
    for (int i = 0; i < topQty; ++i) {
      DocEntryParsed docEntry = queryDocEntries.get(i);
          
      float docWeight = topDocScore.get(i);
      
      // Re-weight all terms using relative document weight
      topDocTerms.clear();
      for (int iDoc = 0; iDoc < docEntry.mWordIds.length; ++iDoc) {
        int wordId = docEntry.mWordIds[iDoc];
        float termScore = docWeight * invTopDocScoreNorm * mSimilObj.getDocTermScore(docEntry, iDoc);
        topDocTerms.add(new IdValPair(wordId, termScore));
      }
    }
    
    // Extract terms with top weights
    topDocTerms.sort(mScoreSorter);
    float topDocTermNorm = 0;
    for (int k = 0; k < Math.min(topDocTerms.size(), mTopTermQty); ++k) {
      IdValPair e = topDocTerms.get(k);
      topDocTermNorm += e.mVal;
    }
    float invTopDocTermNorm = (float) (1.0/Math.max(topDocTermNorm, 1e-9));
    
    for (int k = 0; k < Math.min(topDocTerms.size(), mTopTermQty); ++k) {
      IdValPair e = topDocTerms.get(k);
      
      float score = e.mVal * invTopDocTermNorm;
 
      if (DEBUG_PRINT) 
        System.out.println(String.format("%s %g", mFieldIndex.getWord(e.mId), score));
      
      topTerms.put(e.mId, score);
    }
    
    /*
     * To prevent division by zero if we have no candidate entries or there're no
     * common terms between queries and documents.
     */
    topDocScoreNorm = 1.0f / Math.max(topDocScoreNorm, Float.MIN_NORMAL);

    HashMap<String, DenseVector> res = initResultSet(cands, getFeatureQty());
    
    for (int i = 0; i < docQty; ++i) {
      String docId = cands[i].mDocId;
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(docId);

      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
            
      float rm1score = 0;
      for (int wid : docEntry.mWordIds) {
        Float termScore = topTerms.get(wid);
        if (termScore != null) {
          rm1score += termScore;
        }
      }
      
      float origScore =  mSimilObj.compute(queryEntry, docEntry);
      float finalScore = origScore* mOrigWeight + (1-mOrigWeight) * rm1score;
      
      if (DEBUG_PRINT) 
        System.out.println(String.format("### %g %g -> %g", origScore, rm1score, finalScore));
      
      v.set(0, finalScore);
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
