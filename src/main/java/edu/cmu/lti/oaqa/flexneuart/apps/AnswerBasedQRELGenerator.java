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

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.EvalUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.MiscHelper;


/**
 * <p>A generator of weak QRELs for answer-bearing passages.
 * These QRELs are weak, because every passage containing
 * an answer string is considered to be relevant. However,
 * clearly, some of such passages are spurious. To generate such
 * QRELs, we retrieve a number of passages using Lucene and check
 * if they contain an answer string.</p>
 * 
 * <p>To make it possible,
 * questions and answers passages need to be converted to JSON files.
 * The question JSON-file needs to contain a special extra field
 * with the list of answers. We will then check if an answer
 * appears into one of the retrieved passages. The passage index must be text-raw index.</p>
 * 
 * @author Leonid Boytsov
 *
 */
public class AnswerBasedQRELGenerator {
  final static Logger logger = LoggerFactory.getLogger(AnswerBasedQRELGenerator.class);

  
  public static final class Args {
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mFwdIndexDir;
    
    @Option(name = "-" + CommonParams.QUERY_FILE_PREFIX_PARAM, required = true, usage = "A query file.")
    String mQueryFilePrefix;

    @Option(name = "-" + CommonParams.MAX_NUM_QUERY_PARAM, usage = CommonParams.MAX_NUM_QUERY_DESC)
    int mMaxNumQuery = Integer.MAX_VALUE;
    
    @Option(name = "-" + CommonParams.FIELD_NAME_PARAM, usage = "The field whose text we use to find answers")
    String mFieldName = Const.DEFAULT_QUERY_TEXT_FIELD_NAME;
    
    @Option(name = "-" + CommonParams.COLLECTION_ROOT_DIR_PARAM, usage = CommonParams.COLLECTION_ROOT_DIR_DESC)
    String mCollectRootDir;
    
    @Option(name = "-" + CommonParams.PROVIDER_URI_PARAM, required = true, usage = CommonParams.PROVIDER_URI_DESC)
    String mCandProviderURI;
    
    @Option(name = "-" + CommonParams.CAND_PROVID_PARAM, usage = CommonParams.CAND_PROVID_ADD_CONF_DESC)
    String mCandProviderType = CandidateProvider.CAND_TYPE_LUCENE;
    
    @Option(name = "-" + CommonParams.CAND_PROVID_ADD_CONF_PARAM, usage = CommonParams.CAND_PROVID_ADD_CONF_DESC)
    String mCandProviderConfigName;
    
    @Option(name = "-" + CommonParams.OUTPUT_FILE_PARAM, required = true, usage = CommonParams.OUTPUT_FILE_DESC)
    String mOutFile;
    
    @Option(name = "-" + CommonParams.MAX_CAND_QTY_PARAM, required = true, usage = CommonParams.MAX_CAND_QTY_DESC)
    int mCandQty;
    
    @Option(name = "-" +  CommonParams.THREAD_QTY_PARAM, usage =  CommonParams.THREAD_QTY_DESC)
    int mThreadQty = 1;
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
    
    try {
      out = MiscHelper.createBufferedFileWriter(args.mOutFile);
      
      logger.info("Candidate provider type: " + args.mCandProviderType + " URI: " + args.mCandProviderURI + " config: " + args.mCandProviderConfigName);
      logger.info("Number of threads: " + args.mThreadQty);
      
      ResourceManager resourceManager = new ResourceManager(args.mCollectRootDir, args.mFwdIndexDir, null, null);
      ForwardIndex fwdIndexText = resourceManager.getFwdIndex(args.mFieldName);
      
      if (!fwdIndexText.isTextRaw()) {
        System.out.println("The answer-based QREL-generator works only with raw text indices!");
        System.exit(1);
      }
      
      CandidateProvider  [] candProviders = new CandidateProvider[args.mThreadQty];
      
      candProviders = resourceManager.createCandProviders(args.mCandProviderType, 
                                                            args.mCandProviderURI, 
                                                            args.mCandProviderConfigName, 
                                                            args.mThreadQty);    
      if (candProviders == null) {
        System.err.println("Wrong candidate record provider type: '" + args.mCandProviderType + "'");
        parser.printUsage(System.err);
        System.exit(1);
      }
      
      ArrayList<DataEntryFields> queryArr = DataEntryReader.readParallelQueryData(args.mQueryFilePrefix);

      AnswBasedQRELGenWorker[] workers = new AnswBasedQRELGenWorker[args.mThreadQty];
      
      for (int i = 0; i < args.mThreadQty; ++i) {
        workers[i] = new AnswBasedQRELGenWorker(candProviders[i], args.mFieldName, fwdIndexText, args.mCandQty); 
      }
      
      for (int qnum = 0; qnum < Math.min(queryArr.size(), args.mMaxNumQuery); ++qnum) {
        workers[qnum % args.mThreadQty].addQuery(queryArr.get(qnum), qnum);
      }    
      
      logger.info("Finished loading queries!");
        
      // Start threads
      for (AnswBasedQRELGenWorker e : workers) e.start();
      // Wait till they finish
      for (AnswBasedQRELGenWorker e : workers) e.join(0);  
      
      for (AnswBasedQRELGenWorker e : workers) {
        if (e.isFailure()) {
          System.err.println("At least one thread failed!");
          System.exit(1);
        }
      }
      
      logger.info("Processed " + mQueryQty + " queries");
      
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

  public static synchronized void save(String queryId, ArrayList<String> docIds) throws IOException {
    for (String did : docIds) {
      EvalUtils.saveQrelOneEntry(out, queryId, did, Const.MAX_RELEV_GRADE);
    }
    mQueryQty += 1;
    if (mQueryQty % 100 == 0) {
      logger.info("Processed " + mQueryQty + " queries");
    }
  }
  
  static BufferedWriter out = null;
  static int mQueryQty = 0;
}

class AnswBasedQRELGenWorker extends Thread {
	final static Logger logger = LoggerFactory.getLogger(AnswBasedQRELGenWorker.class);
	
  public AnswBasedQRELGenWorker(CandidateProvider provider, String fieldName, ForwardIndex fwdIndex, int candQty) {
    mFieldName = fieldName;
    mCandProvider = provider;
    mFwdIndex = fwdIndex;
    mCandQty = candQty;
  }
  
  public void addQuery(DataEntryFields e, int queryId) {
    mQueries.add(e);
    mQueryIds.add(queryId);
  }
  
  @Override
  public void run() {

    try {
      ArrayList<String> relDocIds = new ArrayList<String>();
      
      for (int eid = 0; eid < mQueries.size(); ++eid) {
        DataEntryFields queryFields = mQueries.get(eid);         
        
        String queryId = queryFields.mEntryId;
        
        if (queryId == null || queryId.isEmpty()) {
          logger.info("Query ID: " + mQueryIds.get(eid) + " has no entry ID ignoring.");
          continue;
        }
        
        String queryFieldText = queryFields.getString(mFieldName);
        if (queryFieldText == null) {
          queryFieldText = "";
        }
        queryFieldText = queryFieldText.trim();
        if (queryFieldText.isEmpty()) {
          logger.info("Query: " + queryId + " field: " + mFieldName + " is empty, ignoring.");
          continue;
        }

        CandidateInfo cands = mCandProvider.getCandidates(mQueryIds.get(eid), queryFields, mCandQty);
        
        String[] answList = queryFields.getStringArray(Const.ANSWER_LIST_FIELD_NAME);
        if (answList == null || answList.length == 0) {
          logger.info("Query " + queryId + " has no answers, ignoring.");
          continue;
        }
        
        relDocIds.clear();

        for (CandidateEntry e : cands.mEntries) {
          String text = mFwdIndex.getDocEntryTextRaw(e.mDocId);
          if (text == null) {
            logger.warn("No text for doc: " + e.mDocId + 
                        " did you create a positional forward index for the field " + mFieldName);
          }
          text = text.trim().toLowerCase();
          boolean hasAnsw = false; 
          for (String answ : answList) {
            if (answ == null) continue;
            answ = answ.trim().toLowerCase();
            if (text.contains(answ)) {
              hasAnsw = true;
              break;
            }
          }
          if (hasAnsw) {
            relDocIds.add(e.mDocId);
          }
        }
        AnswerBasedQRELGenerator.save(queryId, relDocIds);
      }
    } catch (Exception e) {
      e.printStackTrace();
      mFail=true;
    }
  }
  
  public boolean isFailure() {
    return mFail;
  }
  
  final CandidateProvider mCandProvider;
  final String mFieldName;
  final ForwardIndex mFwdIndex;
  final int mCandQty;
  final ArrayList<DataEntryFields> mQueries = new ArrayList<DataEntryFields>();
  final ArrayList<Integer> mQueryIds = new ArrayList<Integer>();
  boolean mFail = false;
}
