package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;

public class GalagoCandidateProvider extends CandidateProvider {
  final Logger logger = LoggerFactory.getLogger(GalagoCandidateProvider.class);

  @Override
  public boolean isThreadSafe() {
    return true;
  }

  public String getName() {
    return this.getClass().getName();
  }  
  
  public GalagoCandidateProvider(String indexDirName, String galagoOp) throws Exception {
    mParams = Parameters.create();
    mParams.set("scorer", "bm25");
    mParams.set("k", FeatureExtractor.BM25_K1);
    mParams.set("b", FeatureExtractor.BM25_B);
    mParams.set("index", indexDirName);
    
    mGalago = RetrievalFactory.create(mParams);
    mGalagoOp = galagoOp; 
    
    logger.info("Galago operator: " + mGalagoOp);
  }

  @Override
  public CandidateInfo getCandidates(int queryNum, Map<String, String> queryData, int maxQty) throws Exception {
    ArrayList<CandidateEntry> resArr = new ArrayList<CandidateEntry>();
    
    String queryID = queryData.get(ID_FIELD_NAME);
    if (null == queryID) {
      throw new Exception(
          String.format("Query id (%s) is undefined for query # %d",
                        ID_FIELD_NAME, queryNum));
    }        
    
    String text = queryData.get(TEXT_FIELD_NAME);
    if (null == text) {
      throw new Exception(
          String.format("Query (%s) is undefined for query # %d",
                        TEXT_FIELD_NAME, queryNum));
    }
    
    text = text.trim();

    int    numFound = 0;
    
    Node transformed;
    if (!text.isEmpty()) {
      String queryText = String.format("#%s(%s)", mGalagoOp, text.trim());
      Parameters pq = mParams.clone();
      pq.set("requested", maxQty);
      
      // parse and transform query into runnable form
      Node root = StructuredQuery.parse(queryText);
      transformed = mGalago.transformQuery(root, pq);

      // run query
      List<ScoredDocument> results = mGalago.executeQuery(transformed, pq).scoredDocuments;
      
      numFound = results.size();
      
      for (ScoredDocument doc: results) {
        String id = doc.getName();
        float  score = (float) doc.getScore();        
        resArr.add(new CandidateEntry(id, score));
      }
    }    
    
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
    Arrays.sort(results);
        
    return new CandidateInfo(numFound, results);

  }
  
  final private Parameters mParams;
  final private Retrieval  mGalago;
  final private String     mGalagoOp;
}
