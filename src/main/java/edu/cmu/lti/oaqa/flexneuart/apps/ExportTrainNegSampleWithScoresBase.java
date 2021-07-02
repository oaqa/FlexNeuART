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

public abstract class ExportTrainNegSampleWithScoresBase extends ExportTrainNegSampleBase {
  static final Logger logger = LoggerFactory.getLogger(ExportTrainNegSampleWithScoresBase.class);
  
  public ExportTrainNegSampleWithScoresBase(ForwardIndex fwdIndex, 
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
   * @param docs                  documents (with scores) to output
   * @throws Exception
   */
  protected abstract void writeOneEntryData(String queryExportFieldText, 
                                            boolean isTestQuery, 
                                            String queryId, 
                                            HashSet<String> relDocIds, 
                                            ArrayList<CandidateEntry> docs) throws Exception;

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

    ArrayList<CandidateEntry> docs = new ArrayList<CandidateEntry>();

    for (CandidateEntry e : cands.mEntries) {
      docs.add(e);
    }
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleWithScoresBase.class) {
      writeOneEntryData(queryExportFieldText, true /* this is test query */,
                        queryId, relDocIds, docs);
    }
  }

  /**
   *  This version of exportQueryTrain ignores:
   *  1. queries with empty text
   *  2. queries without relevant entries if these entries are not returned by a candidate generator
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

    HashSet<String> allRelDocIds = new HashSet<String>();

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
        allRelDocIds.add(docId);
      }
    }
    if (allRelDocIds.isEmpty()) {
      logger.info("Skipping query " + queryId + " b/c it has no relevant entries.");
      return;
    }
    
    HashSet<String> relDocIds = new HashSet<String>();
    // A list of documents to select negative samples
    ArrayList<String> negPoolDocIds = new ArrayList<String>();

    // Not all the retrieved candidates will be used to select medium-difficulty negatives
    CandidateInfo cands = candProv.getCandidates(queryNum, queryEntry, Math.max(mCandTrainQty, mCandTrain4PosQty));

    for (int k = 0; k < cands.mEntries.length; ++k) {
      CandidateEntry e = cands.mEntries[k];
      
      if (allRelDocIds.contains(e.mDocId)) {
        relDocIds.add(e.mDocId);
      } else {
        if (k < mCandTrainQty) {
          negPoolDocIds.add(e.mDocId);  
        }
      }
    };
    
    if (relDocIds.isEmpty()) {
      logger.info("Skipping query " + queryId + " b/c it has no candidate entries that are relevant.");
      return;
    }
    
    String negPoolDocIdsArr[] = negPoolDocIds.toArray(new String[0]);
    
    HashSet<String> selNegDocIds = new HashSet<String>();
    
    if (mSampleMedNegQty > 0) {
      ArrayList<String> negDocSample = mRandUtils.reservoirSampling(negPoolDocIdsArr, mSampleMedNegQty);
        
      for (String docId : negDocSample) {
        selNegDocIds.add(docId);     
      }
    }
     
    ArrayList<CandidateEntry> docs = new ArrayList<CandidateEntry>();
    
    for (int i = 0; i < cands.mEntries.length; i++) {
      CandidateEntry e = cands.mEntries[i];

      // We generate two types of negative samples. 
      // 1. Include a given number of top candidate entries
      // 2. generate randomly medium-difficulty negative samples from a candidate list.
      //
      // This version does not include easy examples, because they won't have a candidate generator score
      //
      if (i < mHardNegQty ||           // a positive, or a hard negative from the top-K list of candidates
          selNegDocIds.contains(e.mDocId) || // medium difficult negative sampled
          relDocIds.contains(e.mDocId)) { // all relevant
        docs.add(e);
      }
    }
    
    //logger.info("Query #" + queryNum + " relDocIds.size(): " + relDocIds.size() + " docs.size(): " + docs.size());
    
    // Making it thread-safe!
    synchronized (ExportTrainNegSampleWithScoresBase.class) {
      writeOneEntryData(queryExportFieldText, false /* this is train query */,
                        queryId, relDocIds, docs);
    }
  }
}
