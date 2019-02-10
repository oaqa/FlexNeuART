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
  // This function must be thread safe, so make sure
  // it syncs output to the file.
  abstract void exportQuery(int queryNum, 
                            String queryId,
                            String queryText) throws Exception;
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
  private static final String SAMPLE_PAIR_QTY = "sample_pair_qty";
  private static final int SAMPLE_ATTEMPT_QTY = 100; // The number of sampling attempts before we give up
  
  protected ExportTextMatchZoo(LuceneCandidateProvider candProv, ForwardIndex fwdIndex, QrelReader qrels) {
    super(candProv, fwdIndex, qrels);
  }

  static void addOptionDesc(Options opts) {
    opts.addOption(CommonParams.OUTPUT_FILE_PARAM, null, true, CommonParams.OUTPUT_FILE_DESC); 
    opts.addOption(CommonParams.MAX_CAND_QTY_PARAM, null, true, CommonParams.MAX_CAND_QTY_DESC);
    opts.addOption(SAMPLE_PAIR_QTY, null, true, "A number of additional (except QREL) negative samples per query");
  }

  @Override
  String readAddOptions(CommandLine cmd) {
    
    mOutFileName = cmd.getOptionValue(CommonParams.OUTPUT_FILE_PARAM);
    
    if (null == mOutFileName) {
      return "Specify option: " + CommonParams.OUTPUT_FILE_PARAM;
    }
    String tmpn = cmd.getOptionValue(SAMPLE_PAIR_QTY);
    if (null == tmpn) {
      return "Specify option: " + SAMPLE_PAIR_QTY;
    }
    try {
      mSamplePairQty = Integer.parseInt(tmpn);
    } catch (NumberFormatException e) {
      return SAMPLE_PAIR_QTY + " isn't integer: '" + tmpn + "'";
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
  
  // This function assumes that idLeft/textLeft is always more relevant than idRight/textRight
  synchronized void writeField(String idLeft, String textLeft,
                               String idRight, String textRight) throws Exception {
    
    String lineFields[] = { "" + mOutNum, idLeft, textLeft, idRight, textRight, "1"};
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
    
    String lineFields[] = { "", "id_left", "text_left", "id_right", "text_right", "label"};
    mOut.writeNext(lineFields);
    
    mOutNum = 1;
  }

  @Override
  void finishOutput() throws Exception {
    mOut.close();
  }

  @Override
  void exportQuery(int queryNum, String queryId, String queryText) throws Exception {

    HashSet<String> relDocIds = new HashSet<String>();
    HashSet<String> othDocIds = new HashSet<String>();
    HashMap<String, String> queryData = new HashMap<String, String>();

    HashMap<String, String> qq = mQrels.getQueryQrels(queryId);
    
    relDocIds.clear();
    othDocIds.clear();
    
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
    
    queryData.put(CandidateProvider.TEXT_FIELD_NAME, 
                  CandidateProvider.removeAddStopwords(queryText));
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
    
    int relDocIdNum = 0;

    HashSet<String> usedIds = new HashSet<String>();
    
    
    if (relDocIdsArr.length > 0 && othDocIdsArr.length > 0) {
      
      // This method may produce fewer than mSamplePairQty entries in some rare cases
      for (int k = 0; k < SAMPLE_ATTEMPT_QTY * mSamplePairQty; ++k) {
        
        String nonRelDocId = othDocIdsArr[Math.abs(mRandGen.nextInt()) % othDocIdsArr.length];
        String relDocId = relDocIdsArr[relDocIdNum];
        
        String compositeKey = relDocId + "-#-" + nonRelDocId; 
        
        if (usedIds.contains(compositeKey)) {
          //System.out.println("Ignoring repeating combination of keys: " + compositeKey);
          
        } else {
          String relText = CandidateProvider.removeAddStopwords(mFwdIndex.getDocEntryText(relDocId));
          String nonRelText = CandidateProvider.removeAddStopwords(mFwdIndex.getDocEntryText(nonRelDocId));
          if (!relText.isEmpty() && !nonRelText.isEmpty()) {
            writeField(relDocId, relText, nonRelDocId, nonRelText);
            usedIds.add(compositeKey);
            if (usedIds.size() >= mSamplePairQty)
              break;
          }
        }
        
        // To make sure we use as many relevant documents as possible
        // we loop over the array of relevant documents (possibly multiple times)
        relDocIdNum = (relDocIdNum + 1) % relDocIdsArr.length;
      }
    }
  }
  
  CSVWriter             mOut;
  int                   mOutNum;

  int                    mCandQty;
  int                    mSamplePairQty;
  String                 mOutFileName;
  Random                 mRandGen = new Random(0);



}