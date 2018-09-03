package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil.TFIDFSimilarity;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrTFIDFSimilarity extends FeatureExtractor {
  public static String EXTR_TYPE = "TFIDFSimilarity";
  public static String BM25_SIMIL = "bm25";
  public static String K1_PARAM = "k1";
  public static String B_PARAM = "b";
  
  FeatExtrTFIDFSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    // getParam throws an exception if the parameter is not defined
    mFieldName = conf.getReqParamStr(FeatExtrConfig.FIELD_NAME);
    String similType = conf.getReqParamStr(FeatExtrConfig.SIMIL_TYPE);

    mFieldIndex = resMngr.getFwdIndex(mFieldName);

    if (similType.equalsIgnoreCase(BM25_SIMIL))
      mSimilObj = new BM25SimilarityLucene(
                                          conf.getParam(K1_PARAM, BM25SimilarityLucene.DEFAULT_BM25_K1), 
                                          conf.getParam(B_PARAM, BM25SimilarityLucene.DEFAULT_BM25_B), 
                                          mFieldIndex);
    else
      throw new Exception("Unsupported field similarity: " + similType);
 
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = FeatureExtractor.initResultSet(arrDocIds, getFeatureQty());
    
    String query = queryData.get(mFieldName);
    if (null == query) return res;
    query = query.trim();
    if (query.isEmpty()) return res;
    
    DocEntry queryEntry = 
        mFieldIndex.createDocEntry(query.split("\\s+"),
            true  /* True means we generate word ID sequence:
             * in the case of queries, there's never a harm in doing so.
             * If word ID sequence is not used, it will be used only to compute the document length. */              
            );
    
    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      float score = mSimilObj.compute(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      v.set(0, score);      
    }    
    
    return res;
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }

  final String              mFieldName;
  final TFIDFSimilarity     mSimilObj;
  final InMemForwardIndex   mFieldIndex;
}
