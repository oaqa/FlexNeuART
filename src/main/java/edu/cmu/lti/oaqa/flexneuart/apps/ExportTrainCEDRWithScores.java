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
import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.EvalUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.MiscHelper;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

class ExportTrainCEDRWithScores extends ExportTrainNegSampleWithScoresBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainCEDRWithScores.class);
  
  public static final String FORMAT_NAME = "cedr_with_scores";
  
  public static final String PRECISION_FORMAT = "%.6g";
 
  protected ExportTrainCEDRWithScores(ForwardIndex fwdIndex, 
                            String queryExportFieldName, String indexExportFieldName,
                            QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
  }

//Must be called from ExportTrainBase.addAllOptionDesc
  protected static void addOptionsDesc(Options opts) {
    ExportCEDRParams.addOptionsDesc(opts);
  }

  @Override
  protected String readAddOptions(CommandLine cmd) {
    String err = super.readAddOptions(cmd);
    if (!err.isEmpty()) {
      return err;
    }
    
    mTestRunFileName = cmd.getOptionValue(ExportCEDRParams.TEST_RUN_FILE_PARAM);    
    if (null == mTestRunFileName) {
      return "Specify option: " + ExportCEDRParams.TEST_RUN_FILE_PARAM;
    } 
    mDataFileDocsName = cmd.getOptionValue(ExportCEDRParams.DATA_FILE_DOCS_PARAM);
    if (null == mDataFileDocsName) {
      return "Specify option: " + ExportCEDRParams.DATA_FILE_DOCS_PARAM;
    }
    mDataFileQueriesName = cmd.getOptionValue(ExportCEDRParams.DATA_FILE_QUERIES_PARAM);
    if (null == mDataFileQueriesName) {
      return "Specify option: " + ExportCEDRParams.DATA_FILE_QUERIES_PARAM;
    } 
    mQueryDocTrainPairsFileName = cmd.getOptionValue(ExportCEDRParams.QUERY_DOC_PAIR_FILE_PARAM);
    if (null == mQueryDocTrainPairsFileName) {
      return "Specify option: " + ExportCEDRParams.QUERY_DOC_PAIR_FILE_PARAM;
    } 

    return "";
  }
  

  @Override
  protected void startOutput() throws Exception {
    mOutNumDocs = 0;
    mOutNumQueries = 0;
    mOutNumPairs = 0;
    
    mTestRun = MiscHelper.createBufferedFileWriter(mTestRunFileName);
    mDataDocs = MiscHelper.createBufferedFileWriter(mDataFileDocsName);
    mDataQueries = MiscHelper.createBufferedFileWriter(mDataFileQueriesName);
    mQueryDocTrainPairs = MiscHelper.createBufferedFileWriter(mQueryDocTrainPairsFileName);
  }

  @Override
  protected void finishOutput() throws Exception {
    logger.info(String.format("Generated data for %d queries %d documents %d training query-doc pairs",
                                mOutNumQueries, mOutNumDocs, mOutNumPairs));
    
    mTestRun.close();
    mDataDocs.close();
    mDataQueries.close();
    mQueryDocTrainPairs.close();
  }
 
  @Override
  protected void writeOneEntryData(String queryExportFieldText, boolean isTestQuery,
                         String queryId,
                         HashSet<String> relDocIds, 
                         ArrayList<CandidateEntry> docs) throws Exception {
    
   
    if (mSeenQueryIds.contains(queryId)) {
      logger.warn("Ignoring repeating query: " + queryId);
      return;
    }
    mSeenQueryIds.add(queryId);
    mOutNumQueries++;
    
    mDataQueries.write("query\t" + queryId + "\t" + StringUtils.replaceWhiteSpaces(queryExportFieldText) + Const.NL);
    
    int pos = -1;
    for (CandidateEntry e : docs) {
      // We expect the query string to be lower-cased if needed, but document text casing is handled by getDocText
      String docId = e.mDocId;
      String text = getDocText(docId);
      
      if (text == null) {
        logger.warn("Ignoring document " + docId + " b/c of null field");
        continue;
      }
      
      if (text.isEmpty()) {
        logger.warn("Ignoring document " + docId + " b/c of empty field");
        continue;
      }
      
      if (!mSeenDocIds.contains(docId)) {
          // documents can sure repeat, but a data file needs to have only one copy
          text = StringUtils.replaceWhiteSpaces(text);
          if (mMaxWhitespaceTokDocQty > 0) {
            text = StringUtils.truncAtKthWhiteSpaceSeq(text, mMaxWhitespaceTokDocQty);
          }
          mDataDocs.write("doc\t" + docId + "\t" + text + Const.NL);
          mSeenDocIds.add(docId);
          ++mOutNumDocs;
      }
    
      ++pos;
      float score = e.mScore;
      
      if (isTestQuery) {
        EvalUtils.saveTrecOneEntry(mTestRun, 
                                   queryId, docId, 
                                   pos, score, Const.FAKE_RUN_ID); 
      } else {
        mQueryDocTrainPairs.write(queryId + "\t" + 
                                  docId + "\t" + 
                                  String.format(PRECISION_FORMAT, e.mScore) + Const.NL);
      }
      mOutNumPairs++;
    }
  }
  
  int                    mOutNumPairs = 0;
  int                    mOutNumQueries = 0;
  int                    mOutNumDocs = 0;
  
  BufferedWriter         mTestRun;
  BufferedWriter         mDataDocs;
  BufferedWriter         mDataQueries;
  BufferedWriter         mQueryDocTrainPairs;
  
  String                 mTestRunFileName;
  String                 mDataFileDocsName;
  String                 mDataFileQueriesName;
  String                 mQueryDocTrainPairsFileName;
  
  HashSet<String>        mSeenDocIds = new HashSet<>();
  HashSet<String>        mSeenQueryIds = new HashSet<>();
  
}