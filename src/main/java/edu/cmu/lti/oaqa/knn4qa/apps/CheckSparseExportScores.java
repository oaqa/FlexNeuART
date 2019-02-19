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
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Map;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;

import edu.cmu.lti.oaqa.knn4qa.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldSingleScoreFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;
import edu.cmu.lti.oaqa.knn4qa.utils.RandomUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;

import no.uib.cipr.matrix.DenseVector;

/**
 * A class to help check that the inner product of vectors exported NMSLIB 
 * binary sparse space reproduces the original features scores (or 
 * are very close).
 * 
 * @author Leonid Boytsov
 *
 */
public class CheckSparseExportScores {
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
    
    @Option(name = "-model_file", required = true, usage = "Linear-model file used to compute a fusion score.")
    String mLinModelFile;
    
    @Option(name = "-eps_diff", required = true, usage = "The maximum score difference considered to be substantial.")
    float mEpsDiff;
    
    @Option(name = "-verbose", usage = "Print some extra debug info")
    boolean mVerbose;
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
      
      DenseVector compWeights = FeatureExtractor.readFeatureWeights(args.mLinModelFile);
      
      System.out.println("Weights: " + VectorUtils.toString(compWeights));
      
      CompositeFeatureExtractor compositeFeatureExtractor = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);
      
      SingleFieldFeatExtractor[] allExtractors = compositeFeatureExtractor.getCompExtr();    
      int featExtrQty = allExtractors.length;
      SingleFieldSingleScoreFeatExtractor compExtractors[] = new SingleFieldSingleScoreFeatExtractor[featExtrQty];
      
      DenseVector unitWeights = VectorUtils.fill(1, featExtrQty);
      
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
      
      int diffQty = 0, compQty = 0;
        
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
       
        Map<String, DenseVector> res = compositeFeatureExtractor.getFeatures(docIdSample, queryFields);

        for (int i = 0; i < docIdSample.size(); ++i) {
          String docId = docIdSample.get(i);
          
          
          TrulySparseVector queryVect = VectorWrapper.createAnInterleavedFeatureVect(null, queryFields, 
                                                                                    compIndices, compExtractors, 
                                                                                    compWeights);
          
          TrulySparseVector docVect = VectorWrapper.createAnInterleavedFeatureVect(docId, null, 
                                                                                   compIndices, compExtractors, 
                                                                                   unitWeights);
          
          float innerProdVal = TrulySparseVector.scalarProduct(docVect, queryVect);    
          DenseVector featVect = res.get(docId);
          float featBasedVal = (float)compWeights.dot(featVect);
          
          boolean isDiff = Math.abs(innerProdVal - featBasedVal) > args.mEpsDiff;  
          compQty++;
          diffQty += isDiff ? 1 : 0;
          

          
          System.out.println(String.format("Query id: %s Doc id: %s feature-based val: %g inner product val: %g %s",
                                            queryId, docId, featBasedVal, innerProdVal,
                                            isDiff ? "SIGN. DIFF." : ""));
          
          if (args.mVerbose) {
            System.out.println("Weights: "+ VectorUtils.toString(compWeights));
            System.out.println("Features: "+ VectorUtils.toString(featVect));
          }
          
        }
      }
      
      System.out.println(String.format("# of comparisons: %d # of differences: %d", compQty, diffQty));
      if (diffQty != 0) {
        System.err.print("Check failed!");
        System.exit(1);
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
