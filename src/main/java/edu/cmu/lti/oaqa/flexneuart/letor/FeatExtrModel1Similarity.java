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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaOneWordTranRecs;
import edu.cmu.lti.oaqa.flexneuart.giza.TranRecSortByProb;
import edu.cmu.lti.oaqa.flexneuart.resources.JSONKeyValueConfig;
import edu.cmu.lti.oaqa.flexneuart.resources.Model1Data;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.IdValPair;
import edu.cmu.lti.oaqa.flexneuart.utils.IdValParamByValDesc;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;
import net.openhft.koloboke.collect.map.hash.HashIntIntMap;
import net.openhft.koloboke.collect.map.hash.HashIntIntMaps;
import net.openhft.koloboke.collect.map.hash.HashIntObjMap;
import net.openhft.koloboke.collect.map.hash.HashIntObjMaps;
import net.openhft.koloboke.collect.set.hash.HashIntSet;
import net.openhft.koloboke.collect.set.hash.HashIntSets;

public class FeatExtrModel1Similarity extends SingleFieldInnerProdFeatExtractor {
  final static Logger logger = LoggerFactory.getLogger(FeatExtrModel1Similarity.class);
  public static String EXTR_TYPE = "Model1Similarity";
  
  public static String GIZA_ITER_QTY = "gizaIterQty";
  public static String PROB_SELF_TRAN = "probSelfTran";
  public static String MIN_MODEL1_PROB = "minModel1Prob";
  public static String MODEL1_FILE_PREFIX = "model1FilePrefr";
  public static String LAMBDA = "lambda";
  public static String OOV_PROB = "ProbOOV";
  public static String FLIP_DOC_QUERY = "flipDocQuery";
  public static String TOP_TRAN_SCORES_PER_DOCWORD_QTY = "topTranScoresPerDocWordQty";
  public static String TOP_TRAN_CANDWORD_QTY = "topTranCandWordQty";
  private static float MIN_ZERO_LABMDA_TRAN_PROB = 1e-8f;
 
  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
   /*
    * Important note: although this similarity scores should be (approximately) reproducible 
    * by generating query and document vectors with subsequent inner-product computation,
    * this feature seems to be broken and it is not completely clear why. 
    */
  public FeatExtrModel1Similarity(ResourceManager resMngr, JSONKeyValueConfig conf) throws Exception {
    super(resMngr, conf);
   
    mModel1FilePref = conf.getParam(MODEL1_FILE_PREFIX, getIndexFieldName());
    mGizaIterQty = conf.getReqParamInt(GIZA_ITER_QTY);
    mProbSelfTran = conf.getReqParamFloat(PROB_SELF_TRAN);
    mMinModel1Prob = conf.getReqParamFloat(MIN_MODEL1_PROB);
    
    // If these guys aren't default, they can't be set too high, e.g., > 1e6
    // There might be an integer overflow then
    mTopTranScoresPerDocWordQty = conf.getParam(TOP_TRAN_SCORES_PER_DOCWORD_QTY, Integer.MAX_VALUE);
    mTopTranCandWordQty = conf.getParam(TOP_TRAN_CANDWORD_QTY, Integer.MAX_VALUE);

    logger.info("Computing " + mTopTranScoresPerDocWordQty + 
        " top per doc-word scores from top " + mTopTranCandWordQty + 
        " translations per document word");
    
    mLambda = conf.getReqParamFloat(LAMBDA);
    mProbOOV = conf.getParam(OOV_PROB, 1e-9f); 
    
    mFlipDocQuery = conf.getParamBool(FLIP_DOC_QUERY);
    
    mModel1Data = resMngr.getModel1Tran(getIndexFieldName(), 
                                        mModel1FilePref,
                                        false /* no translation table flip */, 
                                        mGizaIterQty, mProbSelfTran, mMinModel1Prob);
    
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    mTopTranCache = HashIntObjMaps.<Integer []>newMutableMap(mModel1Data.mFieldProbTable.length);
  }

  @Override
  public Map<String, DenseVector> getFeatures(CandidateEntry[] cands, DataEntryFields queryFields)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(cands, getFeatureQty()); 
    
    String queryId = queryFields.mEntryId;  
    if (queryId == null) {
      throw new Exception("Undefined query ID!");
    }
    
    DocEntryParsed queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryFields);
    if (queryEntry == null) {
      warnEmptyQueryField(logger, EXTR_TYPE, queryId);
      return res;
    }

    for (CandidateEntry e: cands) {
      DocEntryParsed docEntry = mFieldIndex.getDocEntryParsed(e.mDocId);
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + e.mDocId + "'");
      }  
      
      double score = mFlipDocQuery ? computeOverallScore(docEntry, queryEntry) : computeOverallScore(queryEntry, docEntry);
      
      DenseVector v = res.get(e.mDocId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", e.mDocId));
      }    

      v.set(0, score);
    }  
    
    return res;
  }

  private double [] computeWordScores(int [] wordIds, DocEntryParsed docEntry) throws Exception {
    int queryWordQty = wordIds.length;
    
    double res[] = new double[queryWordQty];
    
    float [] aSourceWordProb = new float[docEntry.mWordIds.length];        
    float sum = 0;    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) 
      sum += docEntry.mQtys[ia];
    
    float invSum = 1/Math.max(1, sum);   
    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) {
      aSourceWordProb[ia] = docEntry.mQtys[ia] * invSum;
    }

    for (int iq=0; iq < queryWordQty;++iq) {
      float totTranProb = 0;
      
      int queryWordId = wordIds[iq];
      
      if (queryWordId >= 0) {        
        for (int ia = 0; ia < docEntry.mWordIds.length; ++ia) {
          int answWordId = docEntry.mWordIds[ia];
          
          float oneTranProb = mModel1Data.mRecorder.getTranProb(answWordId, queryWordId);
          if (answWordId == queryWordId && mProbSelfTran - oneTranProb > Float.MIN_NORMAL) {
            throw new Exception("Bug in re-scaling translation tables: no self-tran probability for: id=" + answWordId + "!");
          }                
          if (oneTranProb >= mMinModel1Prob) {
            totTranProb += oneTranProb * aSourceWordProb[ia];
          }
        }
      }
 
      double collectProb = queryWordId >= 0 ? Math.max(mProbOOV, mModel1Data.mFieldProbTable[queryWordId]) : mProbOOV;
      /* 
         Subtracting log-collection probability adds the same constant factor to each document.
         However, it makes all the scores non-negative and it is also incompatible with the neural
         Model1, which does not use collection smoothing. Hence, for mLambda == 0, we use
         a slightly different formula.
       */
      if (mLambda > Float.MIN_NORMAL) {
        res[iq] = Math.log((1-mLambda)*totTranProb +mLambda*collectProb) - Math.log(mLambda*collectProb);
      } else {
        res[iq] = Math.log(Math.max(MIN_ZERO_LABMDA_TRAN_PROB, totTranProb));
      }
    }
    
    return res;
  }
  
  private double computeOverallScore(DocEntryParsed queryEntry, DocEntryParsed docEntry) throws Exception { 
    double logScore = 0;


    int queryWordQty = queryEntry.mWordIds.length;

    if (mTopTranScoresPerDocWordQty == Integer.MAX_VALUE && 
        mTopTranCandWordQty == Integer.MAX_VALUE) {
      double queryWordScores[] = computeWordScores(queryEntry.mWordIds, docEntry);
      
      for (int iq=0; iq < queryWordQty;++iq) {                                        
        logScore += queryEntry.mQtys[iq] * queryWordScores[iq];
      }
    } else {
      // Map query IDs to QTYs
      HashIntIntMap queryWordIdQtys = HashIntIntMaps.newMutableMap();
      for (int iq=0; iq < queryWordQty;++iq) {   
        queryWordIdQtys.put(queryEntry.mWordIds[iq], queryEntry.mQtys[iq]);
      }
      
      for (IdValPair topIdScore : getTopWordIdsAndScores(docEntry)) {
        int wid = topIdScore.mId;
        if (queryWordIdQtys.containsKey(wid)) {
          logScore +=  queryWordIdQtys.get(wid) * topIdScore.mVal;
        }
      }
    }

    float queryNorm = Math.max(1, queryWordQty);
    
    return logScore / queryNorm;
  }
  
  private ArrayList<IdValPair> getTopWordIdsAndScores(DocEntryParsed doc) throws Exception {
    HashIntSet   wordIdsHash = HashIntSets.newMutableSet();
    
    for (int wid : doc.mWordIds) {
      for (int dstWordId : getTopCandWordIds(wid)) {
        wordIdsHash.add(dstWordId);
      }
    }
    
    int topCandWordIds[] = wordIdsHash.toIntArray();
    double topCandWorIdsScores[] = computeWordScores(topCandWordIds, doc);
    
    ArrayList<IdValPair> res = new ArrayList<IdValPair>();
    
    for (int i = 0; i < topCandWordIds.length; ++i) {
      double score = topCandWorIdsScores[i];
      res.add(new IdValPair(topCandWordIds[i], (float)score));
    }
    
    res.sort(mDescByValComp);

    if (mTopTranScoresPerDocWordQty < Integer.MAX_VALUE) {
    
      int maxQty = doc.mWordIds.length * mTopTranScoresPerDocWordQty;
      
      if (res.size() > maxQty) {
        res.subList(maxQty, res.size()).clear();
      }
      
    }
    /*
    for (int i = 0; i < res.size(); ++i) {
      System.out.println(res.get(i).toString());
    }
    System.out.println("=============");
    */
    
    return res;
   
  }
  
  /**
   * Return words with highest translation scores + the word itself (with respect to a specific word).
   * The result size is at most {@link mTopTranCandWordQty}.
   * This function caches results in a thread-safe fashion.
   * 
   * @param wordId       a word ID
   * 
   * @return an integer array of word IDs.
   */
  private synchronized Integer[] getTopCandWordIds(int wordId) {
    
    if (!mTopTranCache.containsKey(wordId)) {
      
      Integer res [] = {};
      GizaOneWordTranRecs tranRecs = mModel1Data.mRecorder.getTranProbs(wordId);
      
      if (tranRecs != null) {
        
        boolean hasNoSelfTran = true;
        for (int dstWordId : tranRecs.mDstIds) {
          if (dstWordId == wordId) {
            hasNoSelfTran = false;
            break;
          }
        }
        
        TranRecSortByProb tranRecSortedByProb[] = new TranRecSortByProb[tranRecs.mDstIds.length + (hasNoSelfTran ? 1:0)];
        for (int i = 0; i < tranRecs.mDstIds.length; ++i) {
          tranRecSortedByProb[i] = new TranRecSortByProb(tranRecs.mDstIds[i], tranRecs.mProbs[i]);
        }
        if (hasNoSelfTran) {
          tranRecSortedByProb[tranRecs.mDstIds.length] = new TranRecSortByProb(wordId, mProbSelfTran);
        }
        Arrays.sort(tranRecSortedByProb); // Descending by probability
        
        int resQty = Math.min(mTopTranCandWordQty, tranRecSortedByProb.length);
        res = new Integer[resQty];
        for (int i = 0; i < resQty; ++i) {
          res[i] = tranRecSortedByProb[i].mDstWorId; 
        }

      }
      
      mTopTranCache.put(wordId, res);
      return res;
    }
    
    return mTopTranCache.get(wordId);
  }

  /**
   * This feature-generator creates sparse-vector feature representations.
   * 
   */
  @Override
  public boolean isSparse() {
    return true;
  }

  /**
   * Dimensionality is zero, because we generate sparse features.
   * 
   */
  @Override
  public int getDim() {
    return 0;
  }
  
  @Override
  public VectorWrapper getFeatInnerProdQueryVector(DocEntryParsed e) throws Exception {
    if (!mFlipDocQuery) { 
      return getQueryFeatureVectorsForInnerProd(e);
    } else {
      return getDocFeatureVectorsForInnerProd(e);
    }
  }
  
  @Override
  public VectorWrapper getFeatInnerProdDocVector(DocEntryParsed e) throws Exception {
    if (mFlipDocQuery) { 
      return getQueryFeatureVectorsForInnerProd(e);
    } else {
      return getDocFeatureVectorsForInnerProd(e);
    }
  }

  private VectorWrapper getDocFeatureVectorsForInnerProd(DocEntryParsed doc) throws Exception {
    // 1. Get terms with sufficiently high translation probability with
    //    respect to the document
    ArrayList<IdValPair> topIdsScores = getTopWordIdsAndScores(doc);
    
    Collections.sort(topIdsScores); // ascending by ID
    
    int wqty = topIdsScores.size();
    
    TrulySparseVector res = new TrulySparseVector(wqty);
    
    int k = 0;
    for (IdValPair e : topIdsScores) {
      res.mIDs[k] = e.mId;
      res.mVals[k] = e.mVal;
      k++;
    }
    
    return new VectorWrapper(res);
  }

  private VectorWrapper getQueryFeatureVectorsForInnerProd(DocEntryParsed e) {
    int queryWordQty = e.mWordIds.length; 
    
    int nonzWordQty = 0;

    for (int k = 0; k < e.mWordIds.length; ++k) {
      if (e.mWordIds[k] >= 0) {
        nonzWordQty++;
      }
    }
    TrulySparseVector res = new TrulySparseVector(nonzWordQty);
    
    float inv = 1.0f/Math.max(1, queryWordQty);
    
    int idx = 0;
    for (int k = 0; k < queryWordQty; ++k) {
      int wordId = e.mWordIds[k];
      if (wordId >= 0) {
        res.mIDs[idx] = wordId;
        res.mVals[idx] = e.mQtys[k] * inv;
        idx++;
      }
    }
    
    return new VectorWrapper(res);
  }
  
  final ForwardIndex    mFieldIndex;
  final String          mModel1FilePref;
  final Model1Data      mModel1Data;
  final int             mGizaIterQty;
  final float           mProbSelfTran;
  final float           mMinModel1Prob;
  final float           mLambda;
  final float           mProbOOV;
  final boolean         mFlipDocQuery;
  
  final int             mTopTranScoresPerDocWordQty;
  final int             mTopTranCandWordQty;
  
  final IdValParamByValDesc mDescByValComp = new IdValParamByValDesc();
  
  final HashIntObjMap<Integer []> mTopTranCache;
}