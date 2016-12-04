package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
  
  public GalagoCandidateProvider(String indexDirName, String galagoOp, String galagoParams) throws Exception {
    mParams = Parameters.create();
/*
    mParams.set("scorer", "bm25");
    mParams.set("k", FeatureExtractor.BM25_K1);
    mParams.set("b", FeatureExtractor.BM25_B);
*/
    mParams.set("index", indexDirName);
    
    mGalago = RetrievalFactory.create(mParams);
    mGalagoOp = galagoOp; 
    
    logger.info("Galago operator: " + mGalagoOp);
    
    if (galagoParams == null) galagoParams = "";
    Properties prop = new Properties();
    prop.load(new StringReader(galagoParams.replace(',', '\n')));

    // Copy some of the known parameters

    String paramsDouble[] = {// SDM parameters
                             "uniw", // unigram weight
                             "odw",  // ordered window weight
                             "uww",   // unordered window weight
                             // RM parameters
                             "fbOrigWeight", // The weight to give to the original query (default 0.25)
    };
    for (String parName: paramsDouble) {
      String s = prop.getProperty(parName);
      if (s != null) {
        try {
          double v = Double.parseDouble(s);
          mParams.set(parName, v);
        } catch (NumberFormatException e) {
          throw new Exception("Parameter '" + parName + "' value '" + s + "' is not numeric!");
        }
      }
    }
    String paramsInt[] = {// SDM parameters
                          "windowLimit",  // Window proximity limit (default 2)
                          "sdm.od.width", // Window width (default 1)
                          "sdm.uw.width", // Window width (default 4)
                          // RM parameters
                          "fbDocs", // Number of top ranked docs to use in deriving feedback terms (default 20)
                          "fbTerm", // Number of top feedback terms to be added to the query (default 100)
                                    // NOTE: singular "fbTerm" rather than "fbTerms"
    };
    for (String parName: paramsInt) {
      String s = prop.getProperty(parName);
      if (s != null) {
        try {
          int v = Integer.parseInt(s);
          mParams.set(parName, v);
        } catch (NumberFormatException e) {
          throw new Exception("Parameter '" + parName + "' value '" + s + "' is not integer!");
        }
      }
    }
    String paramsStr [] = {
                          // RM parameters
                          "relevanceModel", // org.lemurproject.galago.core.retrieval.prf.RelevanceModel3 by default
    };
    for (String parName: paramsStr) {
      String s = prop.getProperty(parName);
      if (s != null) {
        mParams.set(parName, s);
      }
    }
    logger.info("Galago parameters:\n" + mParams.toString());
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
      logger.info(queryText);
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
