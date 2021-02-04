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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import edu.cmu.lti.oaqa.flexneuart.utils.RandomUtils;

public abstract class ExportTrainNegSampleBase extends ExportTrainBase {
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainNegSampleBase.class);
  
  private final static String MAX_HARD_NEG_QTY              = "hard_neg_qty";
  private final static String MAX_SAMPLE_MEDIUM_NEG_QTY     = "sample_med_neg_qty";
  private final static String MAX_SAMPLE_EASY_NEG_QTY       = "sample_easy_neg_qty";
  
  private final static String MAX_CAND_TRAIN_QTY_PARAM    = "cand_train_qty";
  private final static String MAX_CAND_TRAIN_QTY_DESC = 
                    "A maximum number of candidate records returned by the provider to generate hard negative samples.";
  
  private final static String MAX_CAND_TEST_QTY_PARAM    = "cand_test_qty";
  private final static String MAX_CAND_TEST_QTY_DESC = 
      "A maximum number of candidate records returned by the provider to generate test data.";
  
  
  public static final String MAX_DOC_WHITESPACE_TOK_QTY_PARAM = "max_doc_whitespace_qty";
  public static final String MAX_DOC_WHITESPACE_TOK_QTY_DESC = "max # of whitespace separated tokens to keep in a document";
 
  
  public ExportTrainNegSampleBase(ForwardIndex fwdIndex, 
                                  QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(fwdIndex, qrelsTrain, qrelsTest);
    
    mAllDocIds = fwdIndex.getAllDocIds();
  }

  //Must be called from ExportTrainBase.addAllOptionDesc
  static void addOptionsDesc(Options opts) {
    opts.addOption(MAX_HARD_NEG_QTY, null, true, "A max. # of *HARD* negative examples (all K top-score candidates) per query");
    opts.addOption(MAX_SAMPLE_MEDIUM_NEG_QTY, null, true, "A max. # of *MEDIUM* negative samples (negative candidate and QREL samples) per query");
    opts.addOption(MAX_SAMPLE_EASY_NEG_QTY, null, true, "A max. # of *EASY* negative samples (sampling arbitrary docs) per query");
    
    opts.addOption(MAX_CAND_TRAIN_QTY_PARAM, null, true, MAX_CAND_TRAIN_QTY_DESC);
    opts.addOption(MAX_CAND_TEST_QTY_PARAM, null, true, MAX_CAND_TEST_QTY_DESC);
    opts.addOption(CommonParams.RANDOM_SEED_PARAM, null, true, CommonParams.RANDOM_SEED_DESC);
    
    opts.addOption(MAX_DOC_WHITESPACE_TOK_QTY_PARAM, null, true, MAX_DOC_WHITESPACE_TOK_QTY_DESC);
  }
  
  @Override
  String readAddOptions(CommandLine cmd) {
    
    // Only this sampling parameter should be mandatory
    String tmpn = cmd.getOptionValue(MAX_SAMPLE_MEDIUM_NEG_QTY);
    if (null == tmpn) {
      return "Specify option: " + MAX_SAMPLE_MEDIUM_NEG_QTY;
    }
    try {
      mSampleMedNegQty = Math.max(0, Integer.parseInt(tmpn));
    } catch (NumberFormatException e) {
      return MAX_SAMPLE_MEDIUM_NEG_QTY + " isn't integer: '" + tmpn + "'";
    }
    
    tmpn = cmd.getOptionValue(MAX_HARD_NEG_QTY);
    if (null != tmpn) {
      try {
        mHardNegQty = Math.max(0, Integer.parseInt(tmpn));
      } catch (NumberFormatException e) {
        return MAX_HARD_NEG_QTY + " isn't integer: '" + tmpn + "'";
      }
    }
    
    tmpn = cmd.getOptionValue(MAX_SAMPLE_EASY_NEG_QTY);
    if (null != tmpn) {
      try {
        mSampleEasyNegQty = Math.max(0, Integer.parseInt(tmpn));
      } catch (NumberFormatException e) {
        return MAX_SAMPLE_EASY_NEG_QTY + " isn't integer: '" + tmpn + "'";
      }
    }
    
    logger.info(String.format("# of hard/medium/easy samples per query: %d/%d/%d", mHardNegQty, mSampleMedNegQty, mSampleEasyNegQty));
    
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
    
    logger.info(String.format("# top-scoring training candidates to sample/select from %d", mCandTrainQty));
    logger.info(String.format("# top candidates for validation %d", mCandTestQty));
    
    
    tmpn = cmd.getOptionValue(MAX_DOC_WHITESPACE_TOK_QTY_PARAM);
    if (tmpn != null) {
      try {
        mMaxWhitespaceTokDocQty = Integer.parseInt(tmpn);
      }  catch (NumberFormatException e) {
        return "Maximum number of whitespace document tokens isn't integer: '" + tmpn + "'";
      }
    }
    if (mMaxWhitespaceTokDocQty > 0) {
      logger.info(String.format("Keeping max %d number of whitespace document tokens", mMaxWhitespaceTokDocQty));
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
   * @param candProv       a candidate provider
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
  void exportQuery(CandidateProvider candProv,
                  int queryNum, String queryId, String queryQueryText, String queryFieldText, boolean bIsTestQuery)
      throws Exception {
    if (bIsTestQuery) {
      exportQueryTest(candProv, queryNum, queryId, queryQueryText, queryFieldText);
    } else {
      exportQueryTrain(candProv, queryNum, queryId, queryQueryText, queryFieldText);
    }
  }
  
  void exportQueryTest(CandidateProvider candProv,
                       int queryNum, String queryId, 
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
    
    //queryData.put(Const.TEXT_FIELD_NAME, CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(Const.TEXT_FIELD_NAME, queryQueryText);
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = candProv.getCandidates(queryNum, queryData, mCandTestQty);

    ArrayList<String> docIds = new ArrayList<String>();
    for (CandidateEntry e : cands.mEntries) {
      docIds.add(e.mDocId);
    }
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleBase.class) {
      writeOneEntryData(queryFieldText, true /* this is test query */,
                        queryId, relDocIds, docIds);
    }
  }

  /**
   *  This version of exportQueryTrain ignores queries without relevant entries
   *  as well as queries with empty text.
   */
  void exportQueryTrain(CandidateProvider candProv,
                        int queryNum, String queryId, 
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
    
    //queryData.put(Const.TEXT_FIELD_NAME, CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(Const.TEXT_FIELD_NAME, queryQueryText);
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = candProv.getCandidates(queryNum, queryData, mCandTrainQty);

    // The sampling pool for medium-difficulty items is created from 
    // 1. negative QREL entries
    // 2. candidates retrieved
    for (CandidateEntry e : cands.mEntries) {
      
      if (relDocIds.contains(e.mDocId) || othDocIds.contains(e.mDocId)) {
        continue;
      }
      othDocIds.add(e.mDocId);  
    }
    
    ArrayList<String> allDocIds = new ArrayList<String>(relDocIds);
    
    String othDocIdsArr[] = othDocIds.toArray(new String[0]);
    
    // Second generate three types of negative samples. There's a chance of repeats
    // but they shouldn't be frequent for a reasonable set of parameters. So, let's not
    // bother about this.
    // 1. Include a given number of top candidate entries
    for (int i = 0; i < Math.min(cands.mEntries.length, mHardNegQty); i++) {
      othDocIds.add(cands.mEntries[i].mDocId);
    }
    
    // 2. Generate randomly medium-difficulty negative samples from a candidate list and QRELs:
    //    These are harder than randomly selected (and likely completely non-relevant documents)
    //    but typically easier than a few top candidates, which have highest retrieval scores.
    if (mSampleMedNegQty > 0) {
      ArrayList<String> othDocSample = mRandUtils.reservoirSampling(othDocIdsArr, mSampleMedNegQty);
        
      for (String docId : othDocSample) {
        allDocIds.add(docId);     
      }
    }
    
    // 3. Generate easy negative samples by randomly selecting document IDs from the set of 
    //    all document IDs. These samples are very likely to be:
    //    i) nearly always non-relevant
    //    ii) have very low query-document score, i.e., they should be easy to distinguish from 
    //        a majority of relevant documents, which would have rather large query-document 
    //        similarity scores.
    
    for (int k = 0; k < mSampleEasyNegQty; ++k) {
      int idx = Math.abs(mRandUtils.nextInt()) % mAllDocIds.length;
      String docId = mAllDocIds[idx];
      allDocIds.add(docId);
    }
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleBase.class) {
      writeOneEntryData(queryFieldText, false /* this is train query */,
                        queryId, relDocIds, allDocIds);
    }
  }

  int mHardNegQty = 0;
  int mSampleMedNegQty = 0;
  int mSampleEasyNegQty = 0;
  
  int                    mMaxWhitespaceTokDocQty = -1;
  
  int mCandTrainQty = Integer.MAX_VALUE;
  int mCandTestQty = Integer.MAX_VALUE;
  
  RandomUtils            mRandUtils = null;
  String []              mAllDocIds = null;
}
