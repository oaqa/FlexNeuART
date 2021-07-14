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
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.resources.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.TrulySparseVector;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.RandomUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorUtils;
import no.uib.cipr.matrix.DenseVector;

/**
 * A class to help check that the inner product of vectors exported NMSLIB 
 * binary sparse space reproduces the original features scores (or 
 * are very close). This application is used for debugging scorers.
 * 
 * @author Leonid Boytsov
 *
 */
public class CheckSparseExportScores {
  public static final class Args {
    
    public final static String MAX_NUM_DOC_DESC  = "maximum number of documents to use";
    public final static String MAX_NUM_DOC_PARAM = "max_num_doc"; 
   
    
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mFwdIndexPref;
    
    @Option(name = "-" + CommonParams.MODEL1_ROOT_DIR_PARAM, usage = CommonParams.MODEL1_ROOT_DIR_DESC)
    String mModel1RootDir;
    
    
    @Option(name = "-" + CommonParams.COLLECTION_ROOT_DIR_PARAM, usage = CommonParams.COLLECTION_ROOT_DIR_DESC)
    String mCollectRootDir;
    
    @Option(name = "-" + CommonParams.EMBED_ROOT_DIR_PARAM, usage = CommonParams.EMBED_ROOT_DIR_DESC)
    String mEmbedRootDir;
    
    @Option(name = "-extr_json", required = true, usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PREFIX_PARAM, usage = CommonParams.QUERY_FILE_PREFIX_EXPORT_DESC)
    String mQueryFilePrefix;

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
    
    @Option(name = "-" + CommonParams.BATCH_SIZE_PARAM, usage = CommonParams.BATCH_SIZE_DESC)
    int mBatchSize=16;
  }
  
  public static void main(String argv[]) {
    
    Args args = new Args();
    CmdLineParser parser = null;
    
    RandomUtils rand = new RandomUtils(0);
    
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

      ResourceManager resourceManager = 
          new ResourceManager(args.mCollectRootDir, 
                              args.mFwdIndexPref, 
                              args.mModel1RootDir, 
                              args.mEmbedRootDir);
      
      DenseVector compWeights = FeatureExtractor.readFeatureWeights(args.mLinModelFile);
      
      System.out.println("Weights: " + VectorUtils.toString(compWeights));
      
      CompositeFeatureExtractor compositeFeatureExtractor = resourceManager.getFeatureExtractor(args.mExtrJson);
      
      SingleFieldFeatExtractor[] allExtractors = compositeFeatureExtractor.getCompExtr();    
      int featExtrQty = allExtractors.length;
      SingleFieldInnerProdFeatExtractor compExtractors[] = new SingleFieldInnerProdFeatExtractor[featExtrQty];
      
      DenseVector unitWeights = VectorUtils.fill(1, featExtrQty);
      
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
      
      int diffQty = 0, compQty = 0;
      
      ArrayList<DataEntryFields> queries = DataEntryReader.readParallelQueryData(args.mQueryFilePrefix);

      for (int queryNo = 0;  queryNo < Math.min(args.mMaxNumQuery, queries.size());  ++queryNo) {
        DataEntryFields queryFields = queries.get(queryNo);
        String queryId = queryFields.mEntryId;
       
        Map<String, DenseVector> res = compositeFeatureExtractor.getFeatures(
                          CandidateEntry.createZeroScoreCandListFromDocIds(docIdSample), queryFields);
        

        TrulySparseVector queryVect = VectorUtils.createInterleavedInnerProdQueryFeatVect(queryFields, 
                                                                                          compIndices,
                                                                                          compExtractors, unitWeights);

        for (int batchStart = 0; batchStart < docIdSample.size(); batchStart += args.mBatchSize) {
          int actualBatchQty = Math.min(args.mBatchSize, docIdSample.size() - batchStart);
          
          String docIds[] = new String[actualBatchQty];

          for (int i = 0; i < actualBatchQty; ++i) {
            docIds[i] = docIdSample.get(batchStart + i);
          }
          
          TrulySparseVector docVectArr[] = VectorUtils.createInterleavedInnerProdDocFeatureVecBatch(docIds, compIndices,
                                                                                                    compExtractors, compWeights);
          
          for (int i = 0; i < actualBatchQty; ++i) {
            String did = docIds[i];
            TrulySparseVector docVect = docVectArr[i];

            float innerProdVal = TrulySparseVector.scalarProduct(docVect, queryVect);
            DenseVector featVect = res.get(did);
            float featBasedVal = (float) compWeights.dot(featVect);

            boolean isDiff = Math.abs(innerProdVal - featBasedVal) > args.mEpsDiff;
            compQty++;
            diffQty += isDiff ? 1 : 0;

            System.out.println(String.format("Query id: %s Doc id: %s feature-based val: %g inner product val: %g %s",
                                              queryId, did, featBasedVal, innerProdVal, isDiff ? "SIGN. DIFF." : ""));

            if (args.mVerbose) {
              System.out.println("Weights: " + VectorUtils.toString(compWeights));
              System.out.println("Features: " + VectorUtils.toString(featVect));
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
