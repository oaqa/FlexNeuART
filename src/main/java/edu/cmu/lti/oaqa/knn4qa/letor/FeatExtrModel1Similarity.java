package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrModel1Similarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "Model1Similarity";
  
  public static String GIZA_ITER_QTY = "gizaIterQty";
  public static String PROB_SELF_TRAN = "probSelfTran";
  public static String MIN_MODEL1_PROB = "minModel1Prob";
  public static String MODEL1_SUBDIR = "model1SubDir";
  public static String LAMBDA = "lambda";
  public static String OOV_PROB = "ProbOOV";
  public static String FLIP_DOC_QUERY = "flipDocQuery";
 
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

    mLambda = conf.getReqParamFloat(LAMBDA);
    mProbOOV = conf.getParam(OOV_PROB, 1e-9f); 
    
    mFlipDocQuery = conf.getParamBool(FLIP_DOC_QUERY);
    
    mModel1Data = resMngr.getModel1Tran(mFieldName, 
                                        mModel1SubDir,
                                        false /* no translation table flip */, 
                                        mGizaIterQty, mProbSelfTran, mMinModel1Prob);
    mFieldIndex = resMngr.getFwdIndex(mFieldName);
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
      
      double score = mFlipDocQuery ? computeScore(docEntry, queryEntry) : computeScore(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }    

      v.set(0, score);
    }  
    
    return res;
  }

  private double computeScore(DocEntry queryEntry, DocEntry docEntry) throws Exception { 
    double logScore = 0;
    
    float [] aSourceWordProb = new float[docEntry.mWordIds.length];        
    float sum = 0;    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) 
      sum += docEntry.mQtys[ia];
    
    float invSum = 1/Math.max(1, sum);   
    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) {
      aSourceWordProb[ia] = docEntry.mQtys[ia] * invSum;
    }

    int queryWordQty = queryEntry.mWordIds.length;
    
    for (int iq=0; iq < queryWordQty;++iq) {
      float totTranProb = 0;
      
      int queryWordId = queryEntry.mWordIds[iq];
      int queryRepQty = queryEntry.mQtys[iq];
      
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
                                                    
      logScore += queryRepQty * Math.log((1-mLambda)*totTranProb +mLambda*collectProb);
    }

    float queryNorm = Math.max(1, queryWordQty);
    double logScoreQueryNorm = logScore / queryNorm;
    
    return logScoreQueryNorm;
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }

  final ForwardIndex mFieldIndex;
  final String          mFieldName;
  final String          mModel1SubDir;
  final Model1Data      mModel1Data;
  final int             mGizaIterQty;
  final float           mProbSelfTran;
  final float           mMinModel1Prob;
  final float           mLambda;
  final float           mProbOOV;
  final boolean         mFlipDocQuery;

  /**
   * isSparce, getDim, and getFeatureVectorsForInnerProd
   * need to be fixed.
   * 
   */
  @Override
  public boolean isSparse() {
    // TODO Auto-generated method stub
    return false;
  }

  @Override
  public int getDim() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public VectorWrapper getFeatureVectorsForInnerProd(DocEntry e, boolean isQuery) {
    // TODO Auto-generated method stub
    return null;
  }
}
