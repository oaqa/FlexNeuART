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
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.SingleFieldInnerProdFeatExtractor;
import edu.cmu.lti.oaqa.flexneuart.resources.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.utils.BinReadWriteUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.VectorUtils;

/**
 * A class that exports a number of query and/or document feature vectors to the NMSLIB dense/sparse
 * fusion space (sparse_dense_fusion)
 * 
 * @author Leonid Boytsov
 *
 */
public class ExportToNMSLIBDenseSparseFusion {
  final static Logger logger = LoggerFactory.getLogger(ExportToNMSLIBDenseSparseFusion.class);
  
  public static final class Args {
    
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mFwdIndexDir;
    
    @Option(name = "-" + CommonParams.MODEL1_ROOT_DIR_PARAM, usage = CommonParams.MODEL1_ROOT_DIR_DESC)
    String mModel1RootDir;
    
    @Option(name = "-" + CommonParams.EMBED_ROOT_DIR_PARAM, usage = CommonParams.EMBED_ROOT_DIR_DESC)
    String mEmbedRootDir;
    
    @Option(name = "-" + CommonParams.COLLECTION_ROOT_DIR_PARAM, usage = CommonParams.COLLECTION_ROOT_DIR_DESC)
    String mCollectRootDir;
    
    @Option(name = "-extr_json", required = true, usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PREFIX_PARAM, usage = CommonParams.QUERY_FILE_PREFIX_EXPORT_DESC)
    String mQueryFilePrefix;

    @Option(name = "-out_file", required = true, usage = "Binary output file")
    String mOutFile;
    
    @Option(name = "-" + CommonParams.BATCH_SIZE_PARAM, usage = CommonParams.BATCH_SIZE_DESC)
    int mBatchSize=16;
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
      
      ArrayList<DataEntryFields> queries = null;
      
      if (args.mQueryFilePrefix != null) {
        queries = DataEntryReader.readParallelQueryData(args.mQueryFilePrefix); 
      } 
      
      out = new BufferedOutputStream(new FileOutputStream(args.mOutFile));
      
      ResourceManager resourceManager = 
          new ResourceManager(args.mCollectRootDir, args.mFwdIndexDir, args.mModel1RootDir, args.mEmbedRootDir);
      
      CompositeFeatureExtractor featExtr = resourceManager.getFeatureExtractor(args.mExtrJson);  

      SingleFieldFeatExtractor[] allExtractors = featExtr.getCompExtr();    
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
      
      String[] allDocIds = compIndices[0].getAllDocIds();
      
      int entryQty = queries == null ?  allDocIds.length  : queries.size();
      
      logger.info("Writing the number of entries (" + entryQty + ") to the output file");
      
      out.write(BinReadWriteUtils.intToBytes(entryQty));
      out.write(BinReadWriteUtils.intToBytes(featExtrQty));
      
      for (SingleFieldInnerProdFeatExtractor oneComp : compExtractors) {
        out.write(BinReadWriteUtils.intToBytes(oneComp.isSparse() ? 1 : 0));
        out.write(BinReadWriteUtils.intToBytes(oneComp.getDim()));
      }
      
      if (queries == null) {    
        int docNum = 0;
        
        for (int batchStart = 0; batchStart < allDocIds.length; batchStart += args.mBatchSize) {
          int actualBatchQty = Math.min(args.mBatchSize, allDocIds.length - batchStart);
          
          String docIds[] = new String[actualBatchQty];

          for (int i = 0; i < actualBatchQty; ++i) {
            docIds[i] = allDocIds[batchStart + i];
          }
          
          VectorUtils.writeInnerProdDocVecsBatchToNMSLIBStream(docIds, compIndices, compExtractors, out);
          for (int i = 0; i < actualBatchQty; ++i) {
            ++docNum;
            if (docNum % Const.PROGRESS_REPORT_QTY == 0) {
              logger.info("Exported " + docNum + " docs");
            }
          }
        }

        logger.info("Exported " + docNum + " docs");
      } else {
        int queryQty = 0;
        for (DataEntryFields queryFields : queries) {
          ++queryQty;
          String queryId = queryFields.mEntryId;
          
          if (queryId == null) {
            System.err.println("No query ID: Parsing error, query # " + queryQty);
            System.exit(1);
          }
          
          writeStringId(queryId, out);
          VectorUtils.writeInnerProdQueryVecsToNMSLIBStream(queryFields, compIndices, compExtractors, out);
        }
        logger.info("Exported " + queries.size() + " queries");
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

  private static void writeStringId(String id, BufferedOutputStream out) throws Exception {
    
    // Here we make a fat assumption that the string doesn't contain any non-ascii characters
    if (StringUtils.hasNonAscii(id)) {
      throw new Exception("Invalid id, contains non-ASCII chars: " + id);
    }
    out.write(BinReadWriteUtils.intToBytes(id.length()));
    out.write(id.getBytes());
  }
}
