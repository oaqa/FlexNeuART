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

import java.util.ArrayList;
import java.util.Map;


import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.knn4qa.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.QrelReader;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;

class Worker extends Thread  {
  
  public Worker(ExportTrainDataBase exporter) {
    mExporter = exporter;
  }
  
  public void addQuery(int queryNum, String queryId, 
                       String queryQueryText, String queryFieldText) {
    mQueryNum.add(queryNum);
    mQueryId.add(queryId);
    mQueryQueryText.add(queryQueryText);
    mQueryFieldText.add(queryFieldText);
  }

  @Override
  public void run() {
    for (int i = 0; i < mQueryId.size(); ++i) {
      try {
        mExporter.exportQuery(mQueryNum.get(i), mQueryId.get(i), 
                              mQueryQueryText.get(i), mQueryFieldText.get(i));
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
  
  private ExportTrainDataBase mExporter; 
  private boolean mFail = false;
}

public class ExportTrainPairs {

  private static final String EXPORT_FMT = "export_fmt";
  
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
    
    mOptions.addOption(CommonParams.FWDINDEX_PARAM,         null, true, CommonParams.FWDINDEX_DESC); 
    mOptions.addOption(CommonParams.INDEX_FIELD_NAME_PARAM, null, true, CommonParams.INDEX_FIELD_NAME_DESC);
    mOptions.addOption(EXPORT_FMT,                          null, true, "A type of the export procedure/format");
    mOptions.addOption(CommonParams.QUERY_FILE_PARAM,       null, true, CommonParams.QUERY_FILE_DESC);
    mOptions.addOption(CommonParams.PROVIDER_URI_PARAM,     null, true, CommonParams.LUCENE_INDEX_LOCATION_DESC);
    mOptions.addOption(CommonParams.MAX_NUM_QUERY_PARAM,    null, true, CommonParams.MAX_NUM_QUERY_DESC);
    mOptions.addOption(CommonParams.THREAD_QTY_PARAM,       null, true, CommonParams.THREAD_QTY_DESC);
    mOptions.addOption(CommonParams.QREL_FILE_PARAM,        null, true, CommonParams.QREL_FILE_DESC);
    mOptions.addOption(CommonParams.QUERY_FIELD_NAME_PARAM, null, true, CommonParams.QUERY_FIELD_NAME_DESC);    
    ExportTrainDataBase.addAllOptionDesc(mOptions);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
 
      CommandLine cmd = null;
      int threadQty = 1;
      
      try {
        cmd = parser.parse(mOptions, args);
      } catch (ParseException e) {
        showUsage(e.toString());
      }
      
      String fwdIndex = cmd.getOptionValue(CommonParams.FWDINDEX_PARAM);
      if (fwdIndex == null) {
        showUsageSpecify(CommonParams.FWDINDEX_PARAM);
      }
      String fieldName = cmd.getOptionValue(CommonParams.INDEX_FIELD_NAME_PARAM);
      if (fieldName == null) {
        showUsageSpecify(CommonParams.INDEX_FIELD_NAME_PARAM);
      }
      String exportType = cmd.getOptionValue(EXPORT_FMT);
      if (null == exportType) {
        showUsageSpecify(EXPORT_FMT);
      }
      String qrelFile = cmd.getOptionValue(CommonParams.QREL_FILE_PARAM);
      if (null == qrelFile) {
        showUsageSpecify(CommonParams.QREL_FILE_PARAM);
      }
      QrelReader qrels = new QrelReader(qrelFile);
      
      int maxNumQuery = Integer.MAX_VALUE;
      
      String tmpn = cmd.getOptionValue(CommonParams.MAX_NUM_QUERY_PARAM);
      if (tmpn != null) {
        try {
          maxNumQuery = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Maximum number of queries isn't integer: '" + tmpn + "'");
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
      
      String queryFile = cmd.getOptionValue(CommonParams.QUERY_FILE_PARAM);
      if (null == queryFile) {
        showUsageSpecify(CommonParams.QUERY_FILE_PARAM);
      }
      
      ArrayList<String> queryQueryTexts = new ArrayList<String>();
      ArrayList<String> queryFieldTexts = new ArrayList<String>();
      ArrayList<String> queryIds = new ArrayList<String>();
      
      int queryQty = 0;
      
      Map<String, String> docFields = null;
      
      try (DataEntryReader inp = new DataEntryReader(queryFile)) {
        for (; ((docFields = inp.readNext()) != null) && queryQty < maxNumQuery; ) {

          ++queryQty;
          
          String qid = docFields.get(Const.TAG_DOCNO);
          if (qid == null) {
            System.err.println("Undefined query ID in query # " + queryQty);
            System.exit(1);
          }
          String queryText = CandidateProvider.removeAddStopwords(docFields.get(Const.TEXT_FIELD_NAME));
          String fieldText = CandidateProvider.removeAddStopwords(docFields.get(fieldName));

          if (queryText == null) queryText = "";
          if (fieldText == null) fieldText = "";
          
          if (queryText.isEmpty()) {
            System.out.println(String.format("Ignoring query with empty field '%s' for query '%s'",
                                            Const.TEXT_FIELD_NAME, qid));
            continue;
          }
          if (fieldText.isEmpty()) {
            System.out.println(String.format("Ignoring query with empty field '%s' for query '%s'",
                                            fieldName, qid));
            continue;
          }
          
          queryIds.add(qid);
          queryQueryTexts.add(queryText);
          queryFieldTexts.add(fieldText);
          
        }
      } catch (Exception e1) {
        // TODO Auto-generated catch block
        e1.printStackTrace();
      }

      String providerURI = cmd.getOptionValue(CommonParams.PROVIDER_URI_PARAM);
      if (null == providerURI) {
        showUsageSpecify(CommonParams.PROVIDER_URI_DESC);  
      }
      LuceneCandidateProvider candProv = new LuceneCandidateProvider(providerURI,
                                                                    BM25SimilarityLucene.DEFAULT_BM25_K1, 
                                                                    BM25SimilarityLucene.DEFAULT_BM25_B);
      
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(fwdIndex, null, null);
 
      ExportTrainDataBase oneExport = 
          ExportTrainDataBase.createExporter(exportType, candProv, 
                                             resourceManager.getFwdIndex(fieldName), 
                                             qrels);
      if (null == oneExport) {
        showUsage("Undefined output format: '" + exportType + "'");
      }
      
      String err = oneExport.readAddOptions(cmd);
      if (!err.isEmpty()) {
        showUsage(err);
      }
      oneExport.startOutput();
      
      Worker[] workers = new Worker[threadQty];
      
      for (int threadId = 0; threadId < threadQty; ++threadId) {               
        workers[threadId] = new Worker(oneExport);
      }
      
      int threadId = 0;
      for (int i = 0; i < queryIds.size(); ++i) {
        workers[threadId].addQuery(i, queryIds.get(i), 
                                  queryQueryTexts.get(i), queryFieldTexts.get(i));
        threadId = (threadId + 1) % threadQty;
      }
      
      // Start threads
      for (Worker e : workers) e.start();
      // Wait till they finish
      for (Worker e : workers) e.join(0);  
      
      oneExport.finishOutput();
      
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }
  
  static Options  mOptions = new Options();
  static String   mAppName = "Export training data";
}
