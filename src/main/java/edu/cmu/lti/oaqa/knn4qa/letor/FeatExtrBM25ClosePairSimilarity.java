package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25CloseOrderPairSimilQueryNorm;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25ClosePairSimilarityQueryNormBase;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25CloseUnorderPairSimilQueryNorm;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrBM25ClosePairSimilarity extends SingleFieldFeatExtractor {
  public static String EXTR_TYPE = "BM25ClosePairSimilarity";
  
  public static String QUERY_WINDOW_PARAM = "queryWindow";
  public static String DOC_WINDOW_PARAM = "docWindow";

  public FeatExtrBM25ClosePairSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
    
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    
    mSimilObjs[0] = new BM25CloseOrderPairSimilQueryNorm(
                            conf.getParam(FeatExtrTFIDFSimilarity.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                            conf.getParam(FeatExtrTFIDFSimilarity.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                            conf.getParam(QUERY_WINDOW_PARAM, 4),
                            conf.getParam(DOC_WINDOW_PARAM, 8),
                            mFieldIndex);
    mSimilObjs[1] = new BM25CloseUnorderPairSimilQueryNorm(
                            conf.getParam(FeatExtrTFIDFSimilarity.K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                            conf.getParam(FeatExtrTFIDFSimilarity.B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                            conf.getParam(QUERY_WINDOW_PARAM, 4),
                            conf.getParam(DOC_WINDOW_PARAM, 8),
                            mFieldIndex);
  }

  @Override
  public String getName() {
    return getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData) throws Exception {
    return getSimpleFeatures(arrDocIds, queryData, mFieldIndex, mSimilObjs);
  }

  @Override
  public int getFeatureQty() {
    return 2;
  }

  final BM25ClosePairSimilarityQueryNormBase[]   mSimilObjs = new BM25ClosePairSimilarityQueryNormBase[2];
  final ForwardIndex                             mFieldIndex;
}
