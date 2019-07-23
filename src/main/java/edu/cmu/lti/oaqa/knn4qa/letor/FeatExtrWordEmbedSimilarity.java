package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


import edu.cmu.lti.oaqa.knn4qa.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil_func.AbstractDistance;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLuceneNorm;
import edu.cmu.lti.oaqa.knn4qa.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

public class FeatExtrWordEmbedSimilarity extends SingleFieldInnerProdFeatExtractor {
  public static String EXTR_TYPE = "avgWordEmbed";
  
  public static String QUERY_EMBED_FILE = "queryEmbedFile";
  public static String DOC_EMBED_FILE = "docEmbedFile";
  public static String USE_TFIDF_WEIGHT = "useIDFWeight";
  public static String USE_L2_NORM = "useL2Norm";
  public static String DIST_TYPE = "distType";
  
  FeatExtrWordEmbedSimilarity(FeatExtrResourceManager resMngr, OneFeatExtrConf conf) throws Exception {
    super(resMngr, conf);
 
    mFieldIndex = resMngr.getFwdIndex(getIndexFieldName());
    mSimilObj = new BM25SimilarityLuceneNorm(BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                             BM25SimilarityLucene.DEFAULT_BM25_B, 
                                             mFieldIndex);
    
    mUseIDFWeight = conf.getReqParamBool(USE_TFIDF_WEIGHT);
    mUseL2Norm = conf.getReqParamBool(USE_L2_NORM);
    String distType = conf.getReqParamStr(DIST_TYPE);
    mDist = AbstractDistance.create(distType);
    mIsCosine = distType.compareToIgnoreCase(AbstractDistance.COSINE) == 0;
    
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
  public VectorWrapper getFeatInnerProdVector(DocEntry e, boolean isQuery) throws Exception {
    if (!mIsCosine) {
      throw new Exception("Inner-product representation is available only for the cosine similarity!");
    }
    if (isQuery) {
      return new VectorWrapper(mQueryEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                                                         mUseIDFWeight, 
                                                         true /* normalize vectors!!!*/ ));
    } else {
      return new VectorWrapper(mDocEmbed.getDocAverage(e, mSimilObj, mFieldIndex, 
                                                        mUseIDFWeight, 
                                                        true /* normalize vectors!!!*/ ));
    }
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
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String, DenseVector> res = initResultSet(arrDocIds, getFeatureQty()); 
    DocEntry queryEntry = getQueryEntry(getQueryFieldName(), mFieldIndex, queryData);
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
      // For cosine distance, we add one to convert it into the cosine *SIMILARITY*
      // In such a case, it will be equal to the inner product of the exported normalized
      // dense vectors.
      v.set(0, (mIsCosine ? 1:0) - mDist.compute(queryVect, docVec));
    }
    
    return res;
  }
  
  final ForwardIndex        mFieldIndex;
  final TFIDFSimilarity     mSimilObj;
  final boolean             mUseIDFWeight;
  final boolean             mUseL2Norm;
  final EmbeddingReaderAndRecoder mDocEmbed;
  final EmbeddingReaderAndRecoder mQueryEmbed;
  final AbstractDistance    mDist;
  final boolean             mIsCosine;


}
