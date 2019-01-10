package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.AbstractDistance;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.knn4qa.simil.TFIDFSimilarity;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

public class WordEmbedSimilarity extends FeatureExtractor {
  public static String EXTR_TYPE = "avgWordEmbed";
  
  public static String QUERY_EMBED_FILE = "queryEmbedFile";
  public static String DOC_EMBED_FILE = "docEmbedFile";
  public static String USE_TFIDF_WEIGHT = "useIDFWeight";
  public static String USE_L2_NORM = "useL2Norm";
  public static String DIST_TYPE = "distType";
  
  WordEmbedSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    mFieldName = conf.getReqParamStr(FeatExtrConfig.FIELD_NAME);
    mFieldIndex = resMngr.getFwdIndex(mFieldName);
 
    mSimilObj = new BM25SimilarityLuceneNorm(BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                             BM25SimilarityLucene.DEFAULT_BM25_B, 
                                             mFieldIndex);
    
    String docEmbedFile = conf.getReqParamStr(DOC_EMBED_FILE);
    mDocEmbed = resMngr.getWordEmbed(mFieldName, docEmbedFile);
    String queryEmbedFile = conf.getParam(QUERY_EMBED_FILE, null);
    if (queryEmbedFile == null) {
      mQueryEmbed = mDocEmbed;
    } else {
      mQueryEmbed = resMngr.getWordEmbed(mFieldName, queryEmbedFile);
    }
    mUseIDFWeight = conf.getReqParamBool(USE_TFIDF_WEIGHT);
    mUseL2Norm = conf.getReqParamBool(USE_L2_NORM);
    
    mDist = AbstractDistance.create(conf.getReqParamStr(DIST_TYPE));
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  @Override
  public ArrayList<VectorWrapper> getFeatureVectorsForInnerProd(DocEntry e, boolean isQuery) {
    ArrayList<VectorWrapper> res = new ArrayList<VectorWrapper>();
  
    if (isQuery) {
      res.add(new VectorWrapper(mQueryEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                                                         mUseIDFWeight, mUseL2Norm)));
    } else {
      res.add(new VectorWrapper(mDocEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                                                        mUseIDFWeight, mUseL2Norm)));
    }
    
    return res;
  }
  
  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty()); 
    DocEntry queryEntry = getQueryEntry(mFieldName, mFieldIndex, queryData);
    if (queryEntry == null) return res;
    
    float [] queryVect = mQueryEmbed.getDocAverage(queryEntry, mSimilObj, mFieldIndex, 
                                                   mUseIDFWeight, mUseL2Norm);

    for (String docId : arrDocIds) {
      DocEntry docEntry = mFieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      float [] docVec = mDocEmbed.getDocAverage(docEntry, mSimilObj, mFieldIndex, 
                                                   mUseIDFWeight, mUseL2Norm);

      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      v.set(0, -mDist.compute(queryVect, docVec));
    }
    
    return res;
  }

  @Override
  public int getFeatureQty() {
    return 1;
  }
  
  final ForwardIndex   mFieldIndex;
  final String              mFieldName;
  final TFIDFSimilarity     mSimilObj;
  final boolean             mUseIDFWeight;
  final boolean             mUseL2Norm;
  final EmbeddingReaderAndRecoder mDocEmbed;
  final EmbeddingReaderAndRecoder mQueryEmbed;
  final AbstractDistance    mDist;
}
