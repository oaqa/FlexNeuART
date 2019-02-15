/*
 *  Copyright 2019 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldSingleScoreFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.BinWriteUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.RandomUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A class to help check that the inner product of vectors exported NMSLIB dense/sparse
 * fusion space (sparse_dense_fusion) reproduce the original features scores (or 
 * are very close). This checker works only for *SINGLE-FEATURE* extractor.
 * 
 * @author Leonid Boytsov
 *
 */
public class CheckDenseSparseExportScores {
  public static final class Args {
    
    public final static String MAX_NUM_DOC_DESC  = "maximum number of documents to use";
    public final static String MAX_NUM_DOC_PARAM = "max_num_doc"; 
   
    
    @Option(name = "-" + CommonParams.MEMINDEX_PARAM, required = true, usage = CommonParams.MEMINDEX_DESC)
    String mMemIndexPref;
    
    @Option(name = "-" + CommonParams.GIZA_ROOT_DIR_PARAM, usage = CommonParams.GIZA_ROOT_DIR_DESC)
    String mGizaRootDir;
    
    @Option(name = "-" + CommonParams.EMBED_DIR_PARAM, usage = CommonParams.EMBED_DIR_DESC)
    String mEmbedDir;
    
    @Option(name = "-extr_json", required = true, usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PARAM, required = true, usage = "A query file.")
    String mQueryFile;

    @Option(name = "-" + CommonParams.MAX_NUM_QUERY_PARAM, required = true, usage = CommonParams.MAX_NUM_QUERY_DESC)
    int mMaxNumQuery;
    
    @Option(name = "-" + MAX_NUM_DOC_PARAM, required = true, usage = MAX_NUM_DOC_DESC)
    int mMaxNumDoc;
  }
  
  public static void main(String argv[]) {
    
    Args args = new Args();
    CmdLineParser parser = null;
    
    try {
 
      parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(CommonParams.USAGE_WIDTH));
      parser.parseArgument(argv);
    
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.exit(1);
    }
    
    BufferedOutputStream out = null;
    
    try {

      FeatExtrResourceManager resourceManager = 
          new FeatExtrResourceManager(args.mMemIndexPref, args.mGizaRootDir, args.mEmbedDir);
      
      CompositeFeatureExtractor compositeFeatureExtractor = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);
      
      SingleFieldFeatExtractor[] allExtractors = compositeFeatureExtractor.getCompExtr();    
      int featExtrQty = allExtractors.length;
      SingleFieldSingleScoreFeatExtractor compExtractors[] = new SingleFieldSingleScoreFeatExtractor[featExtrQty];
      
      for (int i = 0; i < featExtrQty; ++i) {
        if (!(allExtractors[i] instanceof SingleFieldSingleScoreFeatExtractor)) {
          System.err.println("Sub-extractor # " + (i+1) + " (" + allExtractors[i].getName() 
              +") doesn't support export to NMSLIB");
          System.exit(1);
        }
        compExtractors[i] = (SingleFieldSingleScoreFeatExtractor)allExtractors[i];
      }

      ForwardIndex              compIndices[] = new ForwardIndex[featExtrQty];
      
      for (int i = 0; i < featExtrQty; ++i) {
        compIndices[i] = resourceManager.getFwdIndex(compExtractors[i].getFieldName());
      }
      
      String allDocIds[] = compIndices[0].getAllDocIds();
      
      ArrayList<String> docIdSample =  RandomUtils.reservoirSampling(allDocIds, args.mMaxNumDoc);
      
      BufferedReader inpQueryBuffer = null;
      
      inpQueryBuffer = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(args.mQueryFile)));
        
      String queryXML = XmlHelper.readNextXMLIndexEntry(inpQueryBuffer);        
        
      for (int queryNo = 0; queryXML!= null && queryNo < args.mMaxNumQuery; 
            queryXML = XmlHelper.readNextXMLIndexEntry(inpQueryBuffer), ++queryNo) {
        
        Map<String, String> queryFields = null;
        // Parse a query
        try {
          queryFields = XmlHelper.parseXMLIndexEntry(queryXML);
        } catch (Exception e) {
          System.err.println("Parsing error, offending DOC:\n" + queryXML);
          System.exit(1);
        }
        
        String queryId = queryFields.get(UtilConst.TAG_DOCNO);
        
        for (int k = 0; k < featExtrQty; ++k) {
          SingleFieldSingleScoreFeatExtractor oneExtr = compExtractors[k];
          ForwardIndex                        oneIndx = compIndices[k];
          String                              fieldName = oneExtr.getFieldName();
          
          Map<String, DenseVector> res = oneExtr.getFeatures(docIdSample, queryFields);
          
          String queryText = queryFields.get(fieldName);
          
          if (queryText == null) {
            System.out.println("No query text, query ID:" + queryId + " field: "+ fieldName);
            queryText = "";
          }
         
          DocEntry queryEntry = oneIndx.createDocEntry(UtilConst.splitOnWhiteSpace(queryText), true); // true means including positions
          
          for (int i = 0; i < docIdSample.size(); ++i) {
            String docId = docIdSample.get(i);
            DocEntry docEntry = oneIndx.getDocEntry(docId);
            
            VectorWrapper docVect = oneExtr.getFeatureVectorsForInnerProd(docEntry, false);         
            VectorWrapper queryVect = oneExtr.getFeatureVectorsForInnerProd(queryEntry, true);
            
            float innerProdVal = VectorWrapper.scalarProduct(docVect, queryVect);
            DenseVector oneFeatVecScore = res.get(docId);
            if (oneFeatVecScore == null) {
              throw new Exception("Bug: no score for " + docId + " extractor: " + oneExtr.getName() + 
                                 " field: " + oneExtr.getFieldName());
            }
            if (oneFeatVecScore.size() != 1) {
              throw new Exception("Bug: feature vector for " + docId + " extractor: " + oneExtr.getName() + 
                  " field: " + oneExtr.getFieldName() + " has size " + oneFeatVecScore.size() + " but we expect size one!");
            }
            float featureVal = (float) oneFeatVecScore.get(0);
            System.out.println(String.format("Query id: %s Doc id: %s field name: %s feature val: %g inner product val: %g extractor: %s ",
                                            queryId, docId, fieldName, featureVal, innerProdVal, oneExtr.getName()));
         }
          
        }
        
      }

      
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Exception while processing: " + e);
      System.exit(1);
    } finally {
      if (out != null) {
        try {
          out.close();
        } catch (IOException e) {
          e.printStackTrace();
          System.exit(1);
        }
      }
    }
  }
  
}