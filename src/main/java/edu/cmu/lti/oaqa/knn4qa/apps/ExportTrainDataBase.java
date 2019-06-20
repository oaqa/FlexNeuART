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

import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.cli.*;

import com.opencsv.CSVWriter;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateInfo;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.QrelReader;


public abstract class ExportTrainDataBase {
  public static final String MATCH_ZOO = "match_zoo";
  
  static ExportTrainDataBase createExporter(String expType,
                                            LuceneCandidateProvider candProv,
                                            ForwardIndex fwdIndex,
                                            QrelReader qrels) {
    if (expType.compareToIgnoreCase(MATCH_ZOO) == 0) {
      return new ExportTextMatchZoo(candProv, fwdIndex, qrels);
    }
    return null;
  }

  static void addAllOptionDesc(Options opts) {
    ExportTextMatchZoo.addOptionDesc(opts);
  }
  // Supposed to return an error message, if some options are missing or poorly formatted
  abstract String readAddOptions(CommandLine cmd);
  /*
   * 1. exportQuery function must be thread-safe: 
   *    make sure the SYNC output to the file.
   * 3. There are potentially two different variants of the query coming from, e.g.
   *    lemmatized and non-lemmatized text fields
   * 4. Specific sub-classes may implement specific functionality.    
   */ 
  abstract void exportQuery(int queryNum, 
                                  String queryId,
                                  String queryQueryText,
                                  String queryFieldText) throws Exception;
  
  abstract void startOutput() throws Exception;
  abstract void finishOutput() throws Exception;

  protected ExportTrainDataBase(LuceneCandidateProvider candProv,
                           ForwardIndex fwdIndex,
                           QrelReader qrels) {
    mCandProv = candProv;
    mFwdIndex = fwdIndex;
    mQrels = qrels;
    
  }

  protected LuceneCandidateProvider   mCandProv;
  protected ForwardIndex              mFwdIndex;
  protected QrelReader                mQrels;

}

class ExportTextMatchZoo extends ExportTrainDataBase {
  private static final String SAMPLE_NEG_QTY = "sample_neg_qty";
  
  protected ExportTextMatchZoo(LuceneCandidateProvider candProv, ForwardIndex fwdIndex, QrelReader qrels) {
    super(candProv, fwdIndex, qrels);
  }

  static void addOptionDesc(Options opts) {
    opts.addOption(CommonParams.OUTPUT_FILE_PARAM, null, true, CommonParams.OUTPUT_FILE_DESC); 
    opts.addOption(CommonParams.MAX_CAND_QTY_PARAM, null, true, CommonParams.MAX_CAND_QTY_DESC);
    opts.addOption(SAMPLE_NEG_QTY, null, true, "A number of negative samples per query or -1 to keep all candidate entries");
  }

  @Override
  String readAddOptions(CommandLine cmd) {
    
    mOutFileName = cmd.getOptionValue(CommonParams.OUTPUT_FILE_PARAM);
    
    if (null == mOutFileName) {
      return "Specify option: " + CommonParams.OUTPUT_FILE_PARAM;
    }
    String tmpn = cmd.getOptionValue(SAMPLE_NEG_QTY);
    if (null == tmpn) {
      return "Specify option: " + SAMPLE_NEG_QTY;
    }
    try {
      mSampleNegQty = Integer.parseInt(tmpn);
    } catch (NumberFormatException e) {
      return SAMPLE_NEG_QTY + " isn't integer: '" + tmpn + "'";
    }
    
    tmpn = cmd.getOptionValue(CommonParams.MAX_CAND_QTY_PARAM);
    if (null == tmpn) {
      return "Specify option: " + CommonParams.MAX_CAND_QTY_PARAM;
    }
    try {
      mCandQty = Integer.parseInt(tmpn);
    } catch (NumberFormatException e) {
      return CommonParams.MAX_CAND_QTY_PARAM + " isn't integer: '" + tmpn + "'";
    }
    return "";
  }
  
  synchronized void writeField(String idLeft, String textLeft,
                               String idRight, String textRight,
                               int relFlag) throws Exception {
    
    String lineFields[] = { idLeft, textLeft, idRight, textRight, "" + relFlag};
    mOut.writeNext(lineFields);
    mOutNum++; 
    
  }
  

  @Override
  void startOutput() throws Exception {
    mOut = new CSVWriter(new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(mOutFileName))),
                        ',', // field separator
                        CSVWriter.NO_QUOTE_CHARACTER, // quote char
                        CSVWriter.NO_ESCAPE_CHARACTER, // escape char
                        UtilConst.NL
                        );
    
    String lineFields[] = {"id_left", "text_left", "id_right", "text_right", "label"};
    mOut.writeNext(lineFields);
    
    mOutNum = 0;
  }

  @Override
  void finishOutput() throws Exception {
    System.out.println("Wrote " + mOutNum + " entries.");
    mOut.close();
  }

  @Override
  void exportQuery(int queryNum, String queryId, 
                   String queryQueryText, String queryFieldText) throws Exception {
    if (mSampleNegQty >= 0) {
      exportQueryTrain(queryNum, queryId, queryQueryText, queryFieldText);
    } else {
      exportQueryTest(queryNum, queryId, queryQueryText, queryFieldText);
    }
  }
  
  void exportQueryTest(int queryNum, String queryId, 
      String queryQueryText, String queryFieldText) throws Exception {
    //System.out.println("Generating test data: just dumping all top-" + mCandQty + " retrieved candidates");
    
    queryFieldText = queryFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryFieldText.isEmpty()) {
      return;
    }
    

    HashSet<String> relDocIds = new HashSet<String>();

    HashMap<String, String> qq = mQrels.getQueryQrels(queryId);
    
    // First just read QRELs
    for (Entry<String, String> e : qq.entrySet()) {
      String docId = e.getKey();
      String label = e.getValue();
      int grade = CandidateProvider.parseRelevLabel(label);
      if (grade >= 1) {
        relDocIds.add(docId);
      }
    }

    HashMap<String, String> queryData = new HashMap<String, String>();
    
    queryData.put(CandidateProvider.TEXT_FIELD_NAME, 
    CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = mCandProv.getCandidates(queryNum, queryData, mCandQty);

    for (CandidateEntry e : cands.mEntries) {
      int relFlag = relDocIds.contains(e.mDocId) ? 1 : 0;
      String docId = e.mDocId;
      
      String text = CandidateProvider.removeAddStopwords(mFwdIndex.getDocEntryText(docId)).trim();
      
      if (!text.isEmpty()) {
        writeField(queryId, queryFieldText, docId, text, relFlag);
      }
    } 
  }

    
  /**
   *  This version of exportQueryTrain ignores queries without relevant entries
   *  as well as queries with empty text.
   */
  void exportQueryTrain(int queryNum, String queryId, 
      String queryQueryText, String queryFieldText) throws Exception {
    
    //System.out.println("Generating training data using noise-contrastive estimation (sampling negative examples)");
    
    queryFieldText = queryFieldText.trim();
    
    // It's super-important to not generate any empty text fields.
    if (queryFieldText.isEmpty()) {
      return;
    }

    HashSet<String> relDocIds = new HashSet<String>();
    HashSet<String> othDocIds = new HashSet<String>();
    HashMap<String, String> queryData = new HashMap<String, String>();

    HashMap<String, String> qq = mQrels.getQueryQrels(queryId);
    
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
      System.out.println("Skipping query " + queryId + " b/c it has no relevant entries.");
      return;
    }
    
    queryData.put(CandidateProvider.TEXT_FIELD_NAME, 
                  CandidateProvider.removeAddStopwords(queryQueryText));
    queryData.put(CandidateProvider.ID_FIELD_NAME, queryId);
    CandidateInfo cands = mCandProv.getCandidates(queryNum, queryData, mCandQty);

    for (CandidateEntry e : cands.mEntries) {
      
      if (relDocIds.contains(e.mDocId) || othDocIds.contains(e.mDocId)) {
        continue;
      }
      othDocIds.add(e.mDocId);
      
    }
    
    String relDocIdsArr [] = relDocIds.toArray(new String[0]);
    String othDocIdsArr[] = othDocIds.toArray(new String[0]);

    
    // First save *ALL* the relevant documents
    for (String docId : relDocIdsArr) {
      String text = CandidateProvider.removeAddStopwords(mFwdIndex.getDocEntryText(docId)).trim();
      
      if (!text.isEmpty()) {
        writeField(queryId, queryFieldText, docId, text, 1);
      }
    }
    
    // Shuffle randomly
    for (int i = othDocIdsArr.length - 1; i >= 1; --i) {
      int k = mRandGen.nextInt(i); // i exclusive
      String tmp = othDocIdsArr[k];
      othDocIdsArr[k] = othDocIdsArr[i];
      othDocIdsArr[i] = tmp;
    }
    
    // Second sample non-relevant ones
    for (int i = 0; i < Math.min(mSampleNegQty, othDocIdsArr.length); ++i) {
      
      String docId = othDocIdsArr[i];
      String text = CandidateProvider.removeAddStopwords(mFwdIndex.getDocEntryText(docId)).trim();
      
      if (text.isEmpty()) {
        continue;
      }
      
      writeField(queryId, queryFieldText, docId, text, 0);
      
    }
    
  }
  
  CSVWriter             mOut;
  int                   mOutNum;

  int                    mCandQty;
  int                    mSampleNegQty;
  String                 mOutFileName;
  Random                 mRandGen = new Random(0);



}