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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.QrelReader;
import edu.cmu.lti.oaqa.knn4qa.utils.RandomUtils;

public abstract class ExportTrainNegSampleBase extends ExportTrainBase {
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainNegSampleBase.class);
  
  private final static String SAMPLE_NEG_QTY = "sample_neg_qty";
  
  private final static String MAX_CAND_TRAIN_QTY_PARAM    = "cand_train_qty";
  private final static String MAX_CAND_TRAIN_QTY_DESC = 
                    "A maximum number of candidate records returned by the provider to generate training data.";
  
  private final static String MAX_CAND_TEST_QTY_PARAM    = "cand_test_qty";
  private final static String MAX_CAND_TEST_QTY_DESC = 
      "A maximum number of candidate records returned by the provider to generate test data.";
  
  public ExportTrainNegSampleBase(LuceneCandidateProvider candProv, ForwardIndex fwdIndex, 
                                  QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(candProv, fwdIndex, qrelsTrain, qrelsTest);
  }

  //Must be called from ExportTrainBase.addAllOptionDesc
  static void addOptionsDesc(Options opts) {
    opts.addOption(SAMPLE_NEG_QTY, null, true, "A number of negative samples per query or -1 to keep all candidate entries");
    
    opts.addOption(MAX_CAND_TRAIN_QTY_PARAM, null, true, MAX_CAND_TRAIN_QTY_DESC);
    opts.addOption(MAX_CAND_TEST_QTY_PARAM, null, true, MAX_CAND_TEST_QTY_DESC);
    opts.addOption(CommonParams.RANDOM_SEED_PARAM, null, true, CommonParams.RANDOM_SEED_DESC);
  }
  
  @Override
  String readAddOptions(CommandLine cmd) {
    
    String tmpn = cmd.getOptionValue(SAMPLE_NEG_QTY);
    if (null == tmpn) {
      return "Specify option: " + SAMPLE_NEG_QTY;
    }
    try {
      mSampleNegQty = Integer.parseInt(tmpn);
    } catch (NumberFormatException e) {
      return SAMPLE_NEG_QTY + " isn't integer: '" + tmpn + "'";
    }
    
    tmpn = cmd.getOptionValue(MAX_CAND_TRAIN_QTY_PARAM);
    if (null != tmpn) {
      try {
        mCandTrainQty = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return MAX_CAND_TRAIN_QTY_PARAM + " isn't integer: '" + tmpn + "'";
      }
    }
    
    int seed = 0;
    tmpn = cmd.getOptionValue(CommonParams.RANDOM_SEED_PARAM);
    if (null != tmpn) {
      try {
        seed = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return CommonParams.RANDOM_SEED_PARAM + " isn't integer: '" + seed + "'";
      }
    }
    
    mRandUtils = new RandomUtils(seed);
     
    tmpn = cmd.getOptionValue(MAX_CAND_TEST_QTY_PARAM);
    if (null != tmpn) {
      try {
        mCandTestQty = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return MAX_CAND_TEST_QTY_PARAM + " isn't integer: '" + tmpn + "'";
      }
    }
    
    return "";
  }
  
  @Override
  abstract void startOutput() throws Exception;

  @Override
  abstract void finishOutput() throws Exception;
  
  /**
   * This is a function to be customized in a child class.
   * 
   * @param queryFieldText a text of the query used for candidate generation
   * @param isTestQuery    true only for test/dev queries
   * @param queryId        query ID
   * @param relDocIds      a hash to check for document relevance
   * @param docIds         documents to output
   * @throws Exception
   */
  abstract void writeOneEntryData(String queryFieldText, boolean isTestQuery, String queryId, 
                                  HashSet<String> relDocIds,
                                  ArrayList<String> docIds) throws Exception;

  @Override
  void exportQuery(int queryNum, String queryId, String queryQueryText, String queryFieldText, boolean bIsTestQuery)
      throws Exception {
    if (bIsTestQuery) {
      exportQueryTest(queryNum, queryId, queryQueryText, queryFieldText);
    } else {
      exportQueryTrain(queryNum, queryId, queryQueryText, queryFieldText);
    }
  }
  
  void exportQueryTest(int queryNum, String queryId, 
                      String queryQueryText, String queryFieldText) throws Exception {
    //logger.info("Generating test data: just dumping all top-" + mCandQty + " retrieved candidates");
    
    queryFieldText = queryFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryFieldText.isEmpty()) {
      return;
    }  

    HashSet<String> relDocIds = new HashSet<String>();

    HashMap<String, String> qq = mQrelsTest.getQueryQrels(queryId);
    
    if (qq == null) {
      logger.info("Skipping query " + queryId + " b/c it has no QREL entries.");
      return;
    }
    
    // First just read QRELs
    for (Entry<String, String> e : qq.entrySet()) {
      String docId = e.getKey();
      String label = e.getValue();
      int grade = CandidateProvider.parseRelevLabel(label);
      if (grade >= 1) {
        relDocIds.add(docId);
      }
    }
    if (relDocIds.isEmpty()) {
      logger.info("Skipping query " + queryId + " b/c it has no relevant entries.");
      return;
    }

    HashMap<String, String> queryData = new HashMap<String, String>();
    
    queryData.put(Const.TEXT_FIELD_NAME, 
    CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = mCandProv.getCandidates(queryNum, queryData, mCandTestQty);

    ArrayList<String> docIds = new ArrayList<String>();
    for (CandidateEntry e : cands.mEntries) {
      docIds.add(e.mDocId);
    } 
    writeOneEntryData(queryFieldText, true /* this is test query */,
                      queryId, relDocIds, docIds);
  }

  /**
   *  This version of exportQueryTrain ignores queries without relevant entries
   *  as well as queries with empty text.
   */
  void exportQueryTrain(int queryNum, String queryId, 
      String queryQueryText, String queryFieldText) throws Exception {
    
    //logger.info("Generating training data using noise-contrastive estimation (sampling negative examples)");
    
    queryFieldText = queryFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryFieldText.isEmpty()) {
      return;
    }

    HashSet<String> relDocIds = new HashSet<String>();
    HashSet<String> othDocIds = new HashSet<String>();
    HashMap<String, String> queryData = new HashMap<String, String>();

    HashMap<String, String> qq = mQrelsTrain.getQueryQrels(queryId);
    
    if (qq == null) {
      logger.info("Skipping query " + queryId + " b/c it has no QREL entries.");
      return;
    }
    
    // First just read QRELs
    for (Entry<String, String> e : qq.entrySet()) {
      String docId = e.getKey();
      String label = e.getValue();
      int grade = CandidateProvider.parseRelevLabel(label);
      if (grade >= 1) {
        relDocIds.add(docId);
      } else {
        othDocIds.add(docId);
      }
    }
    if (relDocIds.isEmpty()) {
      logger.info("Skipping query " + queryId + " b/c it has no relevant entries.");
      return;
    }
    
    queryData.put(Const.TEXT_FIELD_NAME, 
                  CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = mCandProv.getCandidates(queryNum, queryData, mCandTrainQty);

    for (CandidateEntry e : cands.mEntries) {
      
      if (relDocIds.contains(e.mDocId) || othDocIds.contains(e.mDocId)) {
        continue;
      }
      othDocIds.add(e.mDocId);
      
    }
    
    ArrayList<String> allDocIds = new ArrayList<String>(relDocIds);
    
    String othDocIdsArr[] = othDocIds.toArray(new String[0]);
    
    // Second sample non-relevant ones
    ArrayList<String> othDocSample = mRandUtils.reservoirSampling(othDocIdsArr, mSampleNegQty);
      
    for (String docId : othDocSample) {
      allDocIds.add(docId);     
    }
    writeOneEntryData(queryFieldText, false /* this is train query */,
                      queryId, relDocIds, allDocIds);
  }

  int mSampleNegQty = 0;
  int mCandTrainQty = Integer.MAX_VALUE;
  int mCandTestQty = Integer.MAX_VALUE;
  
  RandomUtils            mRandUtils = null;
}
