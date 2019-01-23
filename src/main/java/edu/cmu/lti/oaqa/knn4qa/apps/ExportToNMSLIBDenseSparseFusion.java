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
import java.io.FileOutputStream;
import java.io.IOException;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.knn4qa.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.BinWriteUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;

/**
 * A class that exports a number of query and/or document feature vectors to the NMSLIB dense/sparse
 * fusion space (sparse_dense_fusion)
 * 
 * @author Leonid Boytsov
 *
 */
public class ExportToNMSLIBDenseSparseFusion {
  public static final class Args {
    
    @Option(name = "-" + CommonParams.MEMINDEX_PARAM, required = true, usage = CommonParams.MEMINDEX_DESC)
    String mMemIndexPref;
    
    @Option(name = "-" + CommonParams.GIZA_ROOT_DIR_PARAM, usage = CommonParams.GIZA_ROOT_DIR_DESC)
    String mGizaRootDir;
    
    @Option(name = "-" + CommonParams.EMBED_DIR_PARAM, usage = CommonParams.EMBED_DIR_DESC)
    String mEmbedDir;
    
    @Option(name = "-extr_json",  usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-is_query", usage = "If specified, we generate queries rather than documents.")
    boolean mIsQuery;

    @Option(name = "-out_file", required = true, usage = "Binary output file")
    String mOutFile;
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
      
      out = new BufferedOutputStream(new FileOutputStream(args.mOutFile));
      
      FeatExtrResourceManager resourceManager = 
          new FeatExtrResourceManager(args.mMemIndexPref, args.mGizaRootDir, args.mEmbedDir);
      
      CompositeFeatureExtractor featExtr = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);
      
      /*
       * We will later try to cast this feature extractor reference to a specific
       * feature extractor type (only this type can support export to NMSLIB). 
       */
      SingleFieldFeatExtractor[] compExtractors = featExtr.getCompExtr();
      int featExtrQty = compExtractors.length;

      ForwardIndex              compIndices[] = new ForwardIndex[featExtrQty];
      
      for (int i = 0; i < featExtrQty; ++i) {
        compIndices[i] = resourceManager.getFwdIndex(compExtractors[i].getFieldName());
      }
      
      String[] allDocIds = compIndices[0].getAllDocIds();
      
      out.write(BinWriteUtils.intToBytes(allDocIds.length));
      out.write(BinWriteUtils.intToBytes(featExtrQty));
      
      for (SingleFieldFeatExtractor oneComp : compExtractors) {
        out.write(BinWriteUtils.intToBytes(oneComp.isSparse() ? 1 : 0));
        out.write(BinWriteUtils.intToBytes(oneComp.getDim()));
      }
      
      for (String docId : allDocIds) {
        VectorWrapper.writeAllFeatureVectorsToStream(docId, args.mIsQuery, compIndices, compExtractors, out);
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
