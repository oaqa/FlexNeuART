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

import org.apache.uima.resource.ResourceManager;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.knn4qa.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;

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
    
    @Option(name = "-extr_son",  usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-main_field_name", required = true, usage = "The name of the main field, e.g., text")
    String mMainFieldName;
    
    //ADD OPTION for IS_QUERY
  }
  
  
  
  public static void main(String argv) {
    
    Args args = new Args();
    CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(CommonParams.USAGE_WIDTH));
    
    try {
    
      parser.parseArgument(argv);
    
    } catch (CmdLineException e) {
      
    }
    
    try {
      
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(args.mMemIndexPref, args.mGizaRootDir, args.mEmbedDir);
      
      CompositeFeatureExtractor featExtr = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);
      
      ForwardIndex mainIndex = resourceManager.getFwdIndex(args.mMainFieldName);
      
      for (String docId : mainIndex.getAllDocIds()) {
        //featExtr.getFeatureVectorsForInnerProd();
      }
      
    } catch (Exception e) {
      System.err.println("Exception while processing: " + e);
      System.exit(1);
    }
  }
}
