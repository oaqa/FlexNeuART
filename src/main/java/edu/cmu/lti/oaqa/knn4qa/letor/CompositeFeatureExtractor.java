package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;


/**
 * A composite feature extractor simply combines features from several 
 * component-extractors.
 */
public class CompositeFeatureExtractor extends FeatureExtractor {

  public CompositeFeatureExtractor(FeatExtrResourceManager resMngr, String configFile) throws Exception {
    FeatExtrConfig fullConf = FeatExtrConfig.readConfig(configFile);
    ArrayList<FeatureExtractor> compList = new ArrayList<FeatureExtractor>();
    
    for (OneFeatExtrConf oneExtrConf : fullConf.extractors) {
      FeatureExtractor fe = null;
      String extrType = oneExtrConf.type;
      if (extrType.equalsIgnoreCase(FeatExtrTFIDFSimilarity.EXTR_TYPE)) {
        fe = new FeatExtrTFIDFSimilarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(FeatExtrModel1Similarity.EXTR_TYPE)) {
        fe = new FeatExtrModel1Similarity(resMngr, oneExtrConf);
      } else if (extrType.equalsIgnoreCase(WordEmbedSimilarity.EXTR_TYPE)) {
        fe = new WordEmbedSimilarity(resMngr, oneExtrConf);
      } else 
        throw new Exception("Unsupported extractor type: " + extrType);
      compList.add(fe);
    }
    init(compList);
  }
  
  void init(ArrayList<FeatureExtractor> componentExtractors) {
    int fqty = 0;
    mCompExtr = new FeatureExtractor[componentExtractors.size()];
    for (int i = 0; i < mCompExtr.length; ++i) {
      FeatureExtractor fe = componentExtractors.get(i);
      mCompExtr[i] = fe;
      fqty += fe.getFeatureQty();
    }
    mFeatureQty = fqty;
  }
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds, Map<String, String> queryData)
      throws Exception {
    HashMap<String,DenseVector> res = FeatureExtractor.initResultSet(arrDocIds, getFeatureQty());
    
    int startFeatId = 0;
    for (FeatureExtractor featExtr : mCompExtr) {
      Map<String,DenseVector> subRes = featExtr.getFeatures(arrDocIds, queryData);
      int compQty = featExtr.getFeatureQty();
      for (String docId : arrDocIds) {
        DenseVector dst = res.get(docId);
        DenseVector src = subRes.get(docId);
        for (int fid = 0; fid < compQty; ++fid) {
          dst.set(startFeatId + fid, src.get(fid));
        }
      }
      startFeatId += compQty;
    }
    return res;
  }

  @Override
  public int getFeatureQty() {
    return mFeatureQty;
  }
  
  @Override
  public ArrayList<VectorWrapper> getFeatureVectorsForInnerProd(DocEntry e, boolean isQuery) {
    ArrayList<VectorWrapper> res = new ArrayList<VectorWrapper>();
    for (FeatureExtractor featExtr : mCompExtr) {
      ArrayList<VectorWrapper> tmpRes = featExtr.getFeatureVectorsForInnerProd(e, isQuery);
      res.addAll(tmpRes);
    }
    return res;
  }
  
  private int mFeatureQty;
  private FeatureExtractor[] mCompExtr;

}
