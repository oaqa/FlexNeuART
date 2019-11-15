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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.EvalUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.ExtendedIndexEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.MiscHelper;


/**
 * <p>A generator of weak QRELs for answer-bearing passages.
 * These QRELs are weak, because every passage containing
 * an answer string is considered to be relevant. However,
 * clearly, some of such passages are spurious. To generate such
 * QRELs, we retrieve a number of passages using Lucene and check
 * if they contain an answer string (we look ).</p>
 * 
 * <p>To make it possible,
 * questions and answers passages need to be converted to JSON files.
 * The question JSON-file needs to contain a special extra field
 * with the list of answers. We will then check if an answer
 * appears into one of the retrieved passages. To make matching
 * more robust, the data processing step can do stopping and lemmatization.
 * After such transformation, all tokens are supposed to be separated
 * by a single space.</p>
 * 
 * @author Leonid Boytsov
 *
 */
public class AnswerBasedQrelGenerator {
  final static Logger logger = LoggerFactory.getLogger(AnswerBasedQrelGenerator.class);

  public static final class Args {
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mMemFwdIndex;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PARAM, required = true, usage = "A query file.")
    String mQueryFile;

    @Option(name = "-" + CommonParams.MAX_NUM_QUERY_PARAM, usage = CommonParams.MAX_NUM_QUERY_DESC)
    int mMaxNumQuery = Integer.MAX_VALUE;
    
    @Option(name = "-" + CommonParams.PROVIDER_URI_PARAM, required = true, usage = CommonParams.PROVIDER_URI_DESC)
    String mProviderURI;
    
    @Option(name = "-" + CommonParams.OUTPUT_FILE_PARAM, required = true, usage = CommonParams.OUTPUT_FILE_DESC)
    String mOutFile;
    
    @Option(name = "-" + CommonParams.MAX_CAND_QTY_PARAM, required = true, usage = CommonParams.MAX_CAND_QTY_DESC)
    int mCandQty;
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
    
    BufferedWriter out = null;
    
    try {
      out = MiscHelper.createBufferedFileWriter(args.mOutFile);
      
      LuceneCandidateProvider candProv = 
          new LuceneCandidateProvider(args.mProviderURI,
                                      BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                      BM25SimilarityLucene.DEFAULT_BM25_B);
      
      
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(args.mMemFwdIndex, null, null);
      ForwardIndex fwdIndexText = resourceManager.getFwdIndex(Const.TEXT_FIELD_NAME);
      
      DataEntryReader inp = new DataEntryReader(args.mQueryFile);
      Map<String, String> queryFields = null;  
      ExtendedIndexEntry inpEntry = null;
        
      int queryQty=0;
      
      for (; 
          ((inpEntry = inp.readNextExt()) != null) && queryQty < args.mMaxNumQuery; 
          ++queryQty) {
        
        queryFields = inpEntry.mStringDict;
        
        String queryFieldText = queryFields.get(Const.TEXT_FIELD_NAME);
        if (queryFieldText == null) {
          queryFieldText = "";
        }
        queryFieldText = queryFieldText.trim();
        if (queryFieldText.isEmpty()) {
          logger.info("Query #" + (queryQty + 1) + " is empty, ignoring.");
          continue;
        }
        String queryId = queryFields.get(Const.TAG_DOCNO);
        if (queryId == null || queryId.isEmpty()) {
          logger.info("Query #" + (queryQty + 1) + " has no field: " + Const.TAG_DOCNO + ", ignoring.");
          continue;
        }
        
        CandidateInfo cands = candProv.getCandidates(queryQty, queryFields, args.mCandQty);
        
        ArrayList<String> answList = inpEntry.mStringArrDict.get(Const.ANSWER_LIST_FIELD_NAME);
        if (answList == null || answList.isEmpty()) {
          logger.info("Query #" + (queryQty + 1) + " has no answers, ignoring.");
          continue;
        }
        
        for (CandidateEntry e : cands.mEntries) {
          String text = fwdIndexText.getDocEntryParsedText(e.mDocId);
          if (text == null) {
            logger.warn("No text for doc: " + e.mDocId + 
                " did you create a positional forward index for the field " + Const.TEXT_FIELD_NAME);
          }
          text = " " + text.trim() + " "; // adding sentinels
          boolean hasAnsw = false; 
          for (String answ : answList) {
            if (answ == null) continue;
            answ = answ.trim();
            if (text.contains(answ)) {
              hasAnsw = true;
              break;
            }
          }
          if (hasAnsw) {
            EvalUtils.saveQrelOneEntry(out, queryId, e.mDocId, Const.MAX_RELEV_GRADE);
          }
        } 

        if (queryQty % 100 == 0) logger.info("Processed " + queryQty + " queries");
      }
      logger.info("Processed " + queryQty + " queries");
      inp.close();
      
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
