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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;

public abstract class ExportTrainNegSampleWithoutScoresBase extends ExportTrainNegSampleBase {
  static final Logger logger = LoggerFactory.getLogger(ExportTrainNegSampleWithoutScoresBase.class);
  
  public ExportTrainNegSampleWithoutScoresBase(ForwardIndex fwdIndex, 
                                  String queryExportFieldName, String indexExportFieldName,
                                  QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
  }

  /**
   * This is a function to be customized in a child class.
   * 
   * @param candProv              a candidate provider
   * @param queryExportFieldText  a text of the query to be exported
   * @param isTestQuery           true only for test/dev queries
   * @param queryId               query ID
   * @param relDocIds             a hash to check for document relevance
   * @param docIds                documents to output
   * @throws Exception
   */
  protected abstract void writeOneEntryData(String queryExportFieldText, 
                                            boolean isTestQuery, 
                                            String queryId, 
                                            HashSet<String> relDocIds, 
                                            ArrayList<String> docIds) throws Exception;

  void exportQueryTest(CandidateProvider candProv,
                       int queryNum,
                       DataEntryFields queryEntry,
                       String queryExportFieldText) throws Exception {  
    queryExportFieldText = queryExportFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryExportFieldText.isEmpty()) {
      return;
    }
    
    String queryId = queryEntry.mEntryId;

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
    
    CandidateInfo cands = candProv.getCandidates(queryNum, queryEntry, mCandTestQty);

    ArrayList<String> docIds = new ArrayList<String>();
    for (CandidateEntry e : cands.mEntries) {
      docIds.add(e.mDocId);
    }
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleWithoutScoresBase.class) {
      writeOneEntryData(queryExportFieldText, true /* this is test query */,
                        queryId, relDocIds, docIds);
    }
  }

  /**
   *  This version of exportQueryTrain ignores queries without relevant entries
   *  as well as queries with empty text.
   */
  void exportQueryTrain(CandidateProvider candProv,
                        int queryNum,
                        DataEntryFields queryEntry,
                        String queryExportFieldText) throws Exception {  
    queryExportFieldText = queryExportFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryExportFieldText.isEmpty()) {
      return;
    }

    String queryId = queryEntry.mEntryId;

    HashSet<String> relDocIds = new HashSet<String>();
    // negPoolDocIds is used to sample medium-difficulty negatives
    HashSet<String> negPoolDocIds = new HashSet<String>();

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
        // add to the pool of medium-difficulty negatives
        negPoolDocIds.add(docId);
      }
    }
    if (relDocIds.isEmpty()) {
      logger.info("Skipping query " + queryId + " b/c it has no relevant entries.");
      return;
    }
    
    CandidateInfo cands = candProv.getCandidates(queryNum, queryEntry, mCandTrainQty);

    // The sampling pool for medium-difficulty items is created from 
    // 1. negative QREL entries
    // 2. candidates retrieved
    for (CandidateEntry e : cands.mEntries) {
      if (relDocIds.contains(e.mDocId) || negPoolDocIds.contains(e.mDocId)) {
        continue;
      }
      negPoolDocIds.add(e.mDocId);  
    }

    // Now generate three types of negative samples (hard negative samples aren't randomly selected here!). 
    // There's a chance of repeats  but they shouldn't be frequent for a reasonable set of parameters. 
    // So, let's not worry about this.
    
    // 1. Include a given number of top candidate entries (hard samples, mostly negative)
    ArrayList<String> selDocIds = new ArrayList<String>(relDocIds);
    
    for (int i = 0; i < Math.min(cands.mEntries.length, mHardNegQty); i++) {
      selDocIds.add(cands.mEntries[i].mDocId);
    }
    
    // 2. Generate randomly medium-difficulty negative samples from a candidate list and QRELs:
    //    These are harder than randomly selected documents from all the collection,
    //    but typically easier than a few top candidates, which have highest retrieval scores.

    String negPoolDocIdsArr[] = negPoolDocIds.toArray(new String[0]);
    
    if (mSampleMedNegQty > 0) {
      ArrayList<String> negDocSample = mRandUtils.reservoirSampling(negPoolDocIdsArr, mSampleMedNegQty);
        
      for (String docId : negDocSample) {
        selDocIds.add(docId);     
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
      selDocIds.add(docId);
    }
    
    logger.info("Query #" + queryNum + " relDocIds.size(): " + relDocIds.size() + " selDocIds.size(): " + selDocIds.size());
    
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleWithoutScoresBase.class) {
      writeOneEntryData(queryExportFieldText, false /* this is train query */,
                        queryId, relDocIds, selDocIds);
    }
  }
}
