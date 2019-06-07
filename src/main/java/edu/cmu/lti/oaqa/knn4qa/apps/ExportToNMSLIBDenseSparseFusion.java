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
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.BinWriteUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;
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
    
    @Option(name = "-extr_json", required = true, usage = "A JSON file with a descripton of the extractors")
    String mExtrJson;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PARAM, usage = "If specified, we generate queries rather than documents.")
    String mQueryFile;

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
      
      BufferedReader inpQueryBuffer = null;
      ArrayList<String> queries = null;
      
      if (args.mQueryFile != null) {
        inpQueryBuffer = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(args.mQueryFile)));
      
        queries = new ArrayList<String>();
        
        String queryText = XmlHelper.readNextXMLIndexEntry(inpQueryBuffer);        
        
        for (; queryText!= null; 
            queryText = XmlHelper.readNextXMLIndexEntry(inpQueryBuffer)) {
          queries.add(queryText);
        }
        
        inpQueryBuffer.close();
      }
      
      out = new BufferedOutputStream(new FileOutputStream(args.mOutFile));
      
      FeatExtrResourceManager resourceManager = 
          new FeatExtrResourceManager(args.mMemIndexPref, args.mGizaRootDir, args.mEmbedDir);
      
      CompositeFeatureExtractor featExtr = new CompositeFeatureExtractor(resourceManager, args.mExtrJson);   

      SingleFieldFeatExtractor[] allExtractors = featExtr.getCompExtr();    
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
      
      String[] allDocIds = compIndices[0].getAllDocIds();
      
      int entryQty = queries == null ?  allDocIds.length  : queries.size();
      
      System.out.println("Writing the number of entries (" + entryQty + ") to the output file");
      
      out.write(BinWriteUtils.intToBytes(entryQty));
      out.write(BinWriteUtils.intToBytes(featExtrQty));
      
      for (SingleFieldSingleScoreFeatExtractor oneComp : compExtractors) {
        out.write(BinWriteUtils.intToBytes(oneComp.isSparse() ? 1 : 0));
        out.write(BinWriteUtils.intToBytes(oneComp.getDim()));
      }
      
      if (queries == null) {    
        int docNum = 0;
        for (String docId : allDocIds) {
          writeStringId(docId, out);
          VectorWrapper.writeAllVectorsToNMSLIBStream(docId, null, compIndices, compExtractors, out);
          ++docNum;
          if (docNum % UtilConst.PROGRESS_REPORT_QTY == 0) {
            System.out.println("Exported " + docNum + " docs");
          }
        }
        System.out.println("Exported " + docNum + " docs");
      } else {
        for (String queryText : queries) {
          Map<String, String> queryFields = null;
          // Parse a query
          try {
            queryFields = XmlHelper.parseXMLIndexEntry(queryText);
          } catch (Exception e) {
            System.err.println("Parsing error, offending DOC:\n" + queryText);
            System.exit(1);
          }
          
          String queryId = queryFields.get(UtilConst.TAG_DOCNO);
          
          if (queryId == null) {
            System.err.println("No query ID: Parsing error, offending DOC:\n" + queryText);
            System.exit(1);
          }
          
          writeStringId(queryId, out);
          VectorWrapper.writeAllVectorsToNMSLIBStream(queryId, queryFields, compIndices, compExtractors, out);
        }
        System.out.println("Exported " + queries.size() + " queries");
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
    if (StringUtilsLeo.hasNonAscii(id)) {
      throw new Exception("Invalid id, contains non-ASCII chars: " + id);
    }
    out.write(BinWriteUtils.intToBytes(id.length()));
    out.write(id.getBytes());
  }
}
