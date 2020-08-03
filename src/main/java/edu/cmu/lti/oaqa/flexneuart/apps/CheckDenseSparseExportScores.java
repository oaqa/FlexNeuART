/*
 *  Copyright 2014+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.RandomUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

/**
 * A class to help check that the inner product of vectors exported NMSLIB dense/sparse
 * fusion space (sparse_dense_fusion) reproduce the original features scores (or 
 * are very close). This checker works only for *SINGLE-FEATURE* extractor.
 * This application is used for debugging scorers.
 * 
 * @author Leonid Boytsov
 *
 */
public class CheckDenseSparseExportScores {
  public static final class Args {
    
    public final static String MAX_NUM_DOC_DESC  = "maximum number of documents to use";
    public final static String MAX_NUM_DOC_PARAM = "max_num_doc"; 
   
    
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
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
    
    @Option(name = "-eps_diff", required = true, usage = "The maximum score difference considered to be substantial.")
    float mEpsDiff;
  }
  
  public static void main(String argv[]) {
    
    RandomUtils rand = new RandomUtils(0);
    
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

      int compQty = 0, diffQty = 0;
      
      FeatExtrResourceManager resourceManager = 
          new FeatExtrResourceManager(args.mMemIndexPref, args.mGizaRootDir, args.mEmbedDir);
      
      CompositeFeatureExtractor compositeFeatureExtractor = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);
      
      SingleFieldFeatExtractor[] allExtractors = compositeFeatureExtractor.getCompExtr();    
      int featExtrQty = allExtractors.length;
      SingleFieldInnerProdFeatExtractor compExtractors[] = new SingleFieldInnerProdFeatExtractor[featExtrQty];
      
      for (int i = 0; i < featExtrQty; ++i) {
        if (!(allExtractors[i] instanceof SingleFieldInnerProdFeatExtractor)) {
          System.err.println("Sub-extractor # " + (i+1) + " (" + allExtractors[i].getName() 
              +") doesn't support export to NMSLIB");
          System.exit(1);
        }
        compExtractors[i] = (SingleFieldInnerProdFeatExtractor)allExtractors[i];
      }

      ForwardIndex              compIndices[] = new ForwardIndex[featExtrQty];
      
      for (int i = 0; i < featExtrQty; ++i) {
        compIndices[i] = resourceManager.getFwdIndex(compExtractors[i].getIndexFieldName());
      }
      
      String allDocIds[] = compIndices[0].getAllDocIds();
      
      ArrayList<String> docIdSample =  rand.reservoirSampling(allDocIds, args.mMaxNumDoc);
      
      Map<String, String> queryFields = null;
             
      try (DataEntryReader inp = new DataEntryReader(args.mQueryFile)) {
        for (int queryNo = 0; ((queryFields = inp.readNext()) != null) && queryNo < args.mMaxNumQuery;  ++queryNo) {
          
          String queryId = queryFields.get(Const.TAG_DOCNO);
          
          for (int k = 0; k < featExtrQty; ++k) {
            SingleFieldInnerProdFeatExtractor   oneExtr = compExtractors[k];
            ForwardIndex                        oneIndx = compIndices[k];
            String                              queryFieldName = oneExtr.getQueryFieldName();
            String                              indexFieldName = oneExtr.getIndexFieldName();
            
            Map<String, DenseVector> res = 
                oneExtr.getFeatures(CandidateEntry.createZeroScoreCandListFromDocIds(docIdSample), queryFields);
            
            String queryText = queryFields.get(queryFieldName);
            
            if (queryText == null) {
              System.out.println("No query text, query ID:" + queryId + " query field: "+ queryFieldName);
              queryText = "";
            }
           
            DocEntryParsed queryEntry = oneIndx.createDocEntryParsed(StringUtils.splitOnWhiteSpace(queryText), true); // true means including positions
            
            for (int i = 0; i < docIdSample.size(); ++i) {
              String docId = docIdSample.get(i);
              DocEntryParsed docEntry = oneIndx.getDocEntryParsed(docId);
              
              VectorWrapper docVect = oneExtr.getFeatInnerProdVector(docEntry, false);         
              VectorWrapper queryVect = oneExtr.getFeatInnerProdVector(queryEntry, true);
              
              float innerProdVal = VectorWrapper.scalarProduct(docVect, queryVect);
              DenseVector oneFeatVecScore = res.get(docId);
              if (oneFeatVecScore == null) {
                throw new Exception("Bug: no score for " + docId + " extractor: " + oneExtr.getName() + 
                                   " index field: " + indexFieldName);
              }
              if (oneFeatVecScore.size() != 1) {
                throw new Exception("Bug: feature vector for " + docId + " extractor: " + oneExtr.getName() + 
                    " index field: " + indexFieldName + " has size " + oneFeatVecScore.size() + " but we expect size one!");
              }
              float featureVal = (float) oneFeatVecScore.get(0);
              
              boolean isDiff = Math.abs(innerProdVal - featureVal) > args.mEpsDiff;  
              compQty++;
              diffQty += isDiff ? 1 : 0;
              
              System.out.println(String.format("Query id: %s Doc id: %s field names: %s/%s feature val: %g inner product val: %g extractor: %s %s",
                                              queryId, docId, queryFieldName, indexFieldName, featureVal, innerProdVal, oneExtr.getName(),
                                              isDiff ? "SIGN. DIFF." : ""));
           }
            
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
