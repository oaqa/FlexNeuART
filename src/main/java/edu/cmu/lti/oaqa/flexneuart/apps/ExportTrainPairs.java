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

import java.util.ArrayList;
import java.util.Map;


import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;


/**
 * A wrapper app for generating different types of training data from existing indexed
 * data and QREL (relevance) information. It is a multi-threaded app: We want it 
 * to be as efficient as possible so that we could generate tons of training
 * data in short time.
 * 
 * @author Leonid Boytsov
 *
 */
public class ExportTrainPairs {
  
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainPairs.class);
  private static final String EXPORT_FMT = "export_fmt";
  
  public static final String QREL_FILE_TRAIN_DESC  = "QREL file";
  public static final String QREL_FILE_TRAIN_PARAM = "qrel_file_train";
  
  public static final String QREL_FILE_TEST_DESC  = "QREL file";
  public static final String QREL_FILE_TEST_PARAM = "qrel_file_test";
  
  public static final String QUERY_FILE_TRAIN_DESC = "Training query file";
  public static final String QUERY_FILE_TRAIN_PARAM = "query_file_train";
  
  public static final String QUERY_FILE_TEST_DESC = "Test query file";
  public static final String QUERY_FILE_TEST_PARAM = "query_file_test";
  
  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(mAppName, mOptions);      
    System.exit(1);
  }
  static void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }
  
  /**
   * The class that exports/generate training data for an external LETOR framework such as MatchZoo.
   * It was designed to run efficiently in a multithreaded fashion, b/c potentially we can
   * use it to generate weakly supervised data (using millions of queries).
   */
  public static void main(String[] args) {
    
    mOptions.addOption(CommonParams.INDEX_FIELD_NAME_PARAM, null, true, CommonParams.INDEX_FIELD_NAME_DESC);
    
    mOptions.addOption(EXPORT_FMT,                          null, true, "A type of the export procedure/format");
    
    mOptions.addOption(CommonParams.MAX_NUM_QUERY_TRAIN_PARAM, null, true, CommonParams.MAX_NUM_QUERY_TRAIN_DESC);
    mOptions.addOption(CommonParams.MAX_NUM_QUERY_TEST_PARAM,  null, true, CommonParams.MAX_NUM_QUERY_TEST_DESC);
    
    mOptions.addOption(QUERY_FILE_TRAIN_PARAM, null, true, QUERY_FILE_TRAIN_DESC);
    mOptions.addOption(QUERY_FILE_TEST_PARAM,  null, true, QUERY_FILE_TEST_DESC);
    
    mOptions.addOption(QREL_FILE_TRAIN_PARAM,  null, true, QREL_FILE_TRAIN_DESC);
    mOptions.addOption(QREL_FILE_TEST_PARAM,   null, true, QREL_FILE_TEST_DESC);
    
    mOptions.addOption(CommonParams.PROVIDER_URI_PARAM,     null, true, CommonParams.LUCENE_INDEX_LOCATION_DESC);
    mOptions.addOption(CommonParams.CAND_PROVID_PARAM,      null, true, CandidateProvider.CAND_PROVID_DESC);
    mOptions.addOption(CommonParams.CAND_PROVID_ADD_CONF_PARAM, null, true, CommonParams.CAND_PROVID_ADD_CONF_DESC);
    
    mOptions.addOption(CommonParams.FWDINDEX_PARAM,         null, true, CommonParams.FWDINDEX_DESC); 
    mOptions.addOption(CommonParams.GIZA_ROOT_DIR_PARAM,    null, true,  CommonParams.GIZA_ROOT_DIR_DESC); 
    mOptions.addOption(CommonParams.EMBED_DIR_PARAM,        null, true,  CommonParams.EMBED_DIR_DESC);  

    mOptions.addOption(CommonParams.THREAD_QTY_PARAM,       null, true, CommonParams.THREAD_QTY_DESC);
    
    mOptions.addOption(CommonParams.QUERY_FIELD_NAME_PARAM, null, true, CommonParams.QUERY_FIELD_NAME_DESC); 
       
    ExportTrainBase.addAllOptionDesc(mOptions);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
 
      CommandLine cmd = null;
      int threadQty = 1;
      
      try {
        cmd = parser.parse(mOptions, args);
      } catch (ParseException e) {
        showUsage(e.toString());
      }    

      String providerURI = cmd.getOptionValue(CommonParams.PROVIDER_URI_PARAM);
      if (null == providerURI) {
        showUsageSpecify(CommonParams.PROVIDER_URI_DESC);  
      }
      String candProviderType = cmd.getOptionValue(CommonParams.CAND_PROVID_PARAM);
      if (null == candProviderType) {
        candProviderType = CandidateProvider.CAND_TYPE_LUCENE;
      }
      String candProviderConfigName = cmd.getOptionValue(CommonParams.CAND_PROVID_ADD_CONF_PARAM);
      
      String fwdIndex = cmd.getOptionValue(CommonParams.FWDINDEX_PARAM);
      if (fwdIndex == null) {
        showUsageSpecify(CommonParams.FWDINDEX_PARAM);
      }
      String indexFieldName = cmd.getOptionValue(CommonParams.INDEX_FIELD_NAME_PARAM);
      if (indexFieldName == null) {
        showUsageSpecify(CommonParams.INDEX_FIELD_NAME_PARAM);
      }
      String queryFieldName = cmd.getOptionValue(CommonParams.QUERY_FIELD_NAME_PARAM);
      if (queryFieldName == null) {
        showUsageSpecify(CommonParams.QUERY_FIELD_NAME_PARAM);
      }
      
      String embedDir = cmd.getOptionValue(CommonParams.EMBED_DIR_PARAM);
      String gizaRootDir = cmd.getOptionValue(CommonParams.GIZA_ROOT_DIR_PARAM);

      
      String exportType = cmd.getOptionValue(EXPORT_FMT);
      if (null == exportType) {
        showUsageSpecify(EXPORT_FMT);
      }
      
      String qrelFileTrain = cmd.getOptionValue(QREL_FILE_TRAIN_PARAM);
      if (null == qrelFileTrain) {
        showUsageSpecify(QREL_FILE_TRAIN_PARAM);
      }
      QrelReader qrelsTrain = new QrelReader(qrelFileTrain);
      String qrelFileTest = cmd.getOptionValue(QREL_FILE_TEST_PARAM);
      if (null == qrelFileTest) {
        showUsageSpecify(QREL_FILE_TEST_PARAM);
      }
      QrelReader qrelsTest = new QrelReader(qrelFileTest);
      
      int maxNumQueryTrain = Integer.MAX_VALUE;
      int maxNumQueryTest = Integer.MAX_VALUE;
      
      String tmpn = cmd.getOptionValue(CommonParams.MAX_NUM_QUERY_TRAIN_PARAM);
      if (tmpn != null) {
        try {
          maxNumQueryTrain = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Maximum number of train queries isn't integer: '" + tmpn + "'");
        }
      }
      tmpn = cmd.getOptionValue(CommonParams.MAX_NUM_QUERY_TEST_PARAM);
      if (tmpn != null) {
        try {
          maxNumQueryTest = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Maximum number of test queries isn't integer: '" + tmpn + "'");
        }
      }
           
      tmpn = cmd.getOptionValue(CommonParams.THREAD_QTY_PARAM);
      if (null != tmpn) {
        try {
          threadQty = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Number of threads isn't integer: '" + tmpn + "'");
        }
      }
      
      String queryFileTrain = cmd.getOptionValue(QUERY_FILE_TRAIN_PARAM);
      if (null == queryFileTrain) {
        showUsageSpecify(QUERY_FILE_TRAIN_PARAM);
      }
      String queryFileTest = cmd.getOptionValue(QUERY_FILE_TEST_PARAM);
      if (null == queryFileTest) {
        showUsageSpecify(QUERY_FILE_TEST_PARAM);
      }
  
      
      logger.info("Candidate provider type: " + candProviderType + " URI: " + providerURI + " config: " + candProviderConfigName);
      logger.info("Number of threads: " + threadQty);
      
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(fwdIndex, gizaRootDir, embedDir);
      
      CandidateProvider  [] candProviders = new CandidateProvider[threadQty];
      candProviders = CandidateProvider.createCandProviders(resourceManager, 
                                                            candProviderType, 
                                                            providerURI, 
                                                            candProviderConfigName, threadQty);    
      if (candProviders == null) {
        showUsage("Wrong candidate record provider type: '" + candProviderType + "'");
      }
      
      
      ArrayList<String> queryQueryTexts = new ArrayList<String>();
      ArrayList<String> queryFieldTexts = new ArrayList<String>();
      ArrayList<String> queryIds = new ArrayList<String>();
      ArrayList<Boolean> isTestQuery = new ArrayList<Boolean>();      
      
      Map<String, String> docFields = null;
      
      for (int iQueryType = 0; iQueryType < 2; ++iQueryType) {
        int maxNumQuery = iQueryType == 0 ? maxNumQueryTrain : maxNumQueryTest;
        String queryFile = iQueryType == 0 ? queryFileTrain : queryFileTest;
        DataEntryReader inp = new DataEntryReader(queryFile);
        
        int queryQty = 0; // Reset for each type of queries (train vs. test)
        for (; ((docFields = inp.readNext()) != null) && queryQty < maxNumQuery; ) {

          ++queryQty;
          
          String qid = docFields.get(Const.TAG_DOCNO);
          if (qid == null) {
            logger.error("Undefined query ID in query # " + queryQty);
            System.exit(1);
          }
          
          /*
           *  Thing may seem to be confusing. There is one query field that is used for actual querying.
           *  It is currently hardcoded. There is also the name of the filed in a query JSON entry.
           *  It can be different from the value of the field in the document JSON entry.
           */
          String queryText = docFields.get(Const.TEXT_FIELD_NAME);

          if (queryText == null) queryText = "";
          if (queryText.isEmpty()) {
            logger.info(String.format("Ignoring query with empty field '%s' for query '%s'",
                                      Const.TEXT_FIELD_NAME, qid));
            continue;
          }
          
          String fieldText = docFields.get(queryFieldName);
          
          if (fieldText == null) fieldText = "";
          if (fieldText.isEmpty()) {
            logger.info(String.format("Ignoring query with empty field '%s' for query '%s'",
                                      queryFieldName, qid));
            continue;
          }
          
          queryIds.add(qid);
          queryQueryTexts.add(queryText);
          queryFieldTexts.add(fieldText);
          isTestQuery.add(iQueryType != 0);
        }
        inp.close();
      }
 
      ExportTrainBase oneExport = 
          ExportTrainBase.createExporter(exportType,
                                         resourceManager.getFwdIndex(indexFieldName), 
                                         qrelsTrain, qrelsTest);
      if (null == oneExport) {
        showUsage("Undefined output format: '" + exportType + "'");
      }
      
      String err = oneExport.readAddOptions(cmd);
      if (!err.isEmpty()) {
        showUsage(err);
      }
      oneExport.startOutput();
      
      ExportTrainPairsWorker[] workers = new ExportTrainPairsWorker[threadQty];
      
      for (int threadId = 0; threadId < threadQty; ++threadId) {               
        workers[threadId] = new ExportTrainPairsWorker(candProviders[threadId], oneExport);
      }
      
      int threadId = 0;
      for (int i = 0; i < queryIds.size(); ++i) {
        workers[threadId].addQuery(i, queryIds.get(i), 
                                  queryQueryTexts.get(i), queryFieldTexts.get(i),
                                  isTestQuery.get(i));
        threadId = (threadId + 1) % threadQty;
      }
      
      // Start threads
      for (ExportTrainPairsWorker e : workers) e.start();
      // Wait till they finish
      for (ExportTrainPairsWorker e : workers) e.join(0);  
      
      for (ExportTrainPairsWorker e : workers) {
        if (e.isFailure()) {
          System.err.println("At least one thread failed!");
          System.exit(1);
        }
      }
      
      oneExport.finishOutput();
      
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }
  
  static Options  mOptions = new Options();
  static String   mAppName = "Export training data";
}

class ExportTrainPairsWorker extends Thread  {
  
  public ExportTrainPairsWorker(CandidateProvider candProv,
                                ExportTrainBase exporter) {
    mExporter = exporter;
    mCandProv = candProv;
  }
  
  public void addQuery(int queryNum, String queryId, 
                       String queryQueryText, String queryFieldText,
                       boolean bIsTestQuery) {
    mQueryNum.add(queryNum);
    mQueryId.add(queryId);
    mQueryQueryText.add(queryQueryText);
    mQueryFieldText.add(queryFieldText);
    mIsTestQuery.add(bIsTestQuery);
  }

  @Override
  public void run() {
    for (int i = 0; i < mQueryId.size(); ++i) {
      try {
        mExporter.exportQuery(mCandProv,
                              mQueryNum.get(i), mQueryId.get(i), 
                              mQueryQueryText.get(i), mQueryFieldText.get(i),
                              mIsTestQuery.get(i));
      } catch (Exception e) {
        mFail = true;
        e.printStackTrace();
        break;
      }
    }
  }
  
  public boolean isFailure() {
    return mFail;
  }
  
  ArrayList<String> mQueryId = new ArrayList<String>();
  // mQueryQueryText and mQueryFieldText may come from different fields.
  ArrayList<String> mQueryQueryText = new ArrayList<String>();
  ArrayList<String> mQueryFieldText = new ArrayList<String>();
  ArrayList<Integer> mQueryNum = new ArrayList<Integer>();
  ArrayList<Boolean> mIsTestQuery = new ArrayList<Boolean>();
  
  private CandidateProvider mCandProv;
  private ExportTrainBase mExporter; 
  private boolean mFail = false;
}
