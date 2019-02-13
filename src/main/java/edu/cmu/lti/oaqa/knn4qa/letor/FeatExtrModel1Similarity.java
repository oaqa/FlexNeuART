package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.giza.GizaOneWordTranRecs;
import edu.cmu.lti.oaqa.knn4qa.giza.TranRecSortByProb;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

import net.openhft.koloboke.collect.map.hash.HashIntObjMap;
import net.openhft.koloboke.collect.map.hash.HashIntObjMaps;
import net.openhft.koloboke.collect.set.hash.HashIntSet;
import net.openhft.koloboke.collect.set.hash.HashIntSets;

public class FeatExtrModel1Similarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "Model1Similarity";
  
  public static String GIZA_ITER_QTY = "gizaIterQty";
  public static String PROB_SELF_TRAN = "probSelfTran";
  public static String MIN_MODEL1_PROB = "minModel1Prob";
  public static String MODEL1_SUBDIR = "model1SubDir";
  public static String LAMBDA = "lambda";
  public static String OOV_PROB = "ProbOOV";
  public static String FLIP_DOC_QUERY = "flipDocQuery";
  public static String TOP_TRANQTY = "topTranQty";
 
  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  @Override
  public String getFieldName() {
    return mFieldName;
  }
  
  public FeatExtrModel1Similarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    mFieldName = conf.getReqParamStr(FeatExtrConfig.FIELD_NAME);   
    mModel1SubDir = conf.getParam(MODEL1_SUBDIR, mFieldName);
    mGizaIterQty = conf.getReqParamInt(GIZA_ITER_QTY);
    mProbSelfTran = conf.getReqParamFloat(PROB_SELF_TRAN);
    mMinModel1Prob = conf.getReqParamFloat(MIN_MODEL1_PROB);
    mTopTranQty = conf.getParam(TOP_TRANQTY, 5);

    mLambda = conf.getReqParamFloat(LAMBDA);
    mProbOOV = conf.getParam(OOV_PROB, 1e-9f); 
    
    mFlipDocQuery = conf.getParamBool(FLIP_DOC_QUERY);
    
    mModel1Data = resMngr.getModel1Tran(mFieldName, 
                                        mModel1SubDir,
                                        false /* no translation table flip */, 
                                        mGizaIterQty, mProbSelfTran, mMinModel1Prob);
    
    mFieldIndex = resMngr.getFwdIndex(mFieldName);
    mTopTranCache = HashIntObjMaps.<Integer []>newMutableMap(mModel1Data.mFieldProbTable.length);
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty()); 
    DocEntry queryEntry = getQueryEntry(mFieldName, mFieldIndex, queryData);
    if (queryEntry == null) return res;

    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }  
      
      double score = mFlipDocQuery ? computeOverallScore(docEntry, queryEntry) : computeOverallScore(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }    

      v.set(0, score);
    }  
    
    return res;
  }

  private double [] computeWordScores(int [] wordIds, DocEntry docEntry) throws Exception {
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
                                                    
      res[iq] = Math.log((1-mLambda)*totTranProb +mLambda*collectProb);
    }
    
    return res;
  }
  
  private double computeOverallScore(DocEntry queryEntry, DocEntry docEntry) throws Exception { 
    double logScore = 0;
    
    double queryWordScores[] = computeWordScores(queryEntry.mWordIds, docEntry);
    
    int queryWordQty = queryEntry.mWordIds.length;
    
    for (int iq=0; iq < queryWordQty;++iq) {                                        
      logScore += queryEntry.mQtys[iq] * queryWordScores[iq];
    }

    float queryNorm = Math.max(1, queryWordQty);
    
    return logScore / queryNorm;
  }
  
  /**
   * Return words with highest translation scores +
   * the word itself.
   * 
   * @param wordId  a word ID
   * @return an integer array of word IDs.
   */
  private synchronized Integer[] getTopWordIds(int wordId) {
    
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
        
        int resQty = Math.min(mTopTranQty, tranRecSortedByProb.length);
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

  @Override
  public int getFeatureQty() {
    return 1;
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
  public VectorWrapper getFeatureVectorsForInnerProd(DocEntry e, boolean isQuery) throws Exception {

    if (isQuery) {
      return getQueryFeatureVectorsForInnerProd(e);
    } else {
      return getDocFeatureVectorsForInnerProd(e);
    }
 
  }

  private VectorWrapper getDocFeatureVectorsForInnerProd(DocEntry e) throws Exception {
    // 1. Get terms with sufficiently high translation probability
    HashIntSet   allAddWordIdsHash = HashIntSets.newMutableSet();
    
    for (int wordId : e.mWordIds) {
      for (int dstWordId : getTopWordIds(wordId)) {
        allAddWordIdsHash.add(dstWordId);
      }
    }
    
    int wqty = allAddWordIdsHash.size();
    TrulySparseVector res = new TrulySparseVector(wqty);
    int k = 0;
    for (int wordId : allAddWordIdsHash) {
      res.mIDs[k++] = wordId;
    }
    
    Arrays.sort(res.mIDs);
    
    double scores[] = computeWordScores(res.mIDs, e);
    
    // 2. Compute their respective translation scores
    for (k = 0; k < wqty; ++k) {
      res.mVals[k] = (float)scores[k];
    }
    
    return new VectorWrapper(res);
  }

  private VectorWrapper getQueryFeatureVectorsForInnerProd(DocEntry e) {
    int queryWordQty = e.mWordIds.length; 
    int nonzWordQty = 0;
    int totFreq = 0;
    for (int k = 0; k < e.mWordIds.length; ++k) {
      if (e.mWordIds[k] >= 0) {
        nonzWordQty++;
        totFreq += e.mQtys[k];
      }
    }
    TrulySparseVector res = new TrulySparseVector(nonzWordQty);
    
    float inv = 1.0f/Math.max(1, totFreq);
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
  final String          mFieldName;
  final String          mModel1SubDir;
  final Model1Data      mModel1Data;
  final int             mGizaIterQty;
  final float           mProbSelfTran;
  final float           mMinModel1Prob;
  final float           mLambda;
  final float           mProbOOV;
  final boolean         mFlipDocQuery;
  final int             mTopTranQty;
  final HashIntObjMap<Integer []> mTopTranCache;
}
