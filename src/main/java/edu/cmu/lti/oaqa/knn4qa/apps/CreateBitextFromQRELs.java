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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.QrelReader;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtils;

/**
 * Creating parallel (bi-text) corpus from relevance judgements, queries,
 * and indexed documents.
 *
 * Importantly, it requires a forward index to store sequences of words
 * rather than merely bag-of-words entries.
 * 
 * @author Leonid Boytsov
 *
 */
public class CreateBitextFromQRELs {
  
  public static final class Args {  
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mFwdIndex;
    @Option(name = "-" + CommonParams.QUERY_FILE_PARAM, required = true, usage = CommonParams.QUERY_FILE_DESC)
    String mQueryFile;
    
    @Option(name = "-" + CommonParams.INDEX_FIELD_NAME_PARAM, required = true, usage = CommonParams.INDEX_FIELD_NAME_DESC)
    String mIndexField;
    @Option(name = "-" + CommonParams.QUERY_FIELD_NAME_PARAM, required = true, usage = CommonParams.INDEX_FIELD_NAME_DESC)
    String mQueryField;
    @Option(name = "-" + CommonParams.QREL_FILE_PARAM, required = true, usage = CommonParams.QREL_FILE_DESC)
    String mQrelFile;
    @Option(name = "-output_dir", required = true, usage = "bi-text output directory")
    String mOutDir;
    
    @Option(name = "-" + CommonParams.MAX_NUM_QUERY_PARAM, required = false, usage = CommonParams.MAX_NUM_QUERY_DESC)
    int mMaxNumQuery = Integer.MAX_VALUE;
    
    @Option(name = "-max_doc_query_qty_ratio", required = true, usage = "Max. ratio of # words in docs to # of words in queries")
    float mDocQueryWordRatio;
    /*
    @Option(name = "-sample_words", required=false, usage = "If specified, document words are sampled.")
    boolean mSample;
    @Option(name = "-sample_qty", required=false, usage = "Number of times sampling is repeated. " +
                                                          "If this parameter is set to, e.g., 2, " +
                                                          " we obtain 2*n bi-text entries, where n is the number of queries.")
    */
    int mSampleQty = 1;
  }
  
  public static void main(String[] argv) {

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
    
    
    try {
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(args.mFwdIndex, null, null);
      
      String fieldName = args.mIndexField;
      
      ForwardIndex fwdIndex = resourceManager.getFwdIndex(fieldName);
      
      QrelReader qrels = new QrelReader(args.mQrelFile);
      
      try (BufferedWriter questFile = 
          new BufferedWriter(new FileWriter(args.mOutDir + Const.PATH_SEP + Const.BITEXT_QUEST_PREFIX + fieldName))) {
        try (BufferedWriter answFile = 
            new BufferedWriter(new FileWriter(args.mOutDir + Const.PATH_SEP + Const.BITEXT_ANSW_PREFIX + fieldName))) {
          
          Map<String, String> docFields = null;
          int queryQty = 0;
          
          try (DataEntryReader inp = new DataEntryReader(args.mQueryFile)) {
            for (; ((docFields = inp.readNext()) != null) && queryQty < args.mMaxNumQuery; ) {

              ++queryQty;
              
              String qid = docFields.get(Const.TAG_DOCNO);
              if (qid == null) {
                System.err.println("Undefined query ID in query # " + queryQty);
                System.exit(1);
              }
              
              String queryText = docFields.get(args.mQueryField);
              if (queryText == null) queryText = "";
              queryText = queryText.trim();
              String [] queryWords = StringUtils.splitOnWhiteSpace(queryText);
            
              if (queryText.isEmpty() || queryWords.length == 0) {
                System.out.println("Empty text in query # " + queryQty + " ignoring");
                continue;
              }
              
              float queryWordQtyInv = 1.0f / queryWords.length;
              
              HashMap<String, String> relInfo = qrels.getQueryQrels(qid);
              for (Entry<String, String> e : relInfo.entrySet()) {
                String did = e.getKey();
                int grade = CandidateProvider.parseRelevLabel(e.getValue());
                if (grade >= 1) {
                  DocEntry dentry = fwdIndex.getDocEntry(did);
                  if (dentry.mWordIdSeq == null) {
                    System.err.println("Index for the field " + fieldName + " doesn't have words sequences!");
                    System.exit(1);
                  }      

                  ArrayList<String> answWords = new ArrayList<String>();
                  /*
                   * In principle, queries can be longer then documents, especially,
                   * when we write the "tail" part of the documents. However, the 
                   * difference should not be large and longer queries will be 
                   * truncated by  
                   */
                  for (int wid : dentry.mWordIdSeq) {
                    if (answWords.size() * queryWordQtyInv >= args.mDocQueryWordRatio) {
                      questFile.write(queryText + Const.NL);
                      answFile.write(StringUtils.joinWithSpace(answWords) + Const.NL);
                      answWords.clear();
                    }
                    
                    if (wid >=0) { // -1 is OOV
                      answWords.add(fwdIndex.getWord(wid));
                    }
                  }
                  
                  if (!answWords.isEmpty()) {
                    questFile.write(queryText + Const.NL);
                    answFile.write(StringUtils.joinWithSpace(answWords) + Const.NL);
                  }
                  
                }
              }
            }
          }
          
        }
      }
    } catch (Exception e) {
      System.err.println(e.getMessage());
      System.exit(1);
    }
    
    
  }

}
