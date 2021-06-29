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

import org.apache.commons.cli.*;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;


public abstract class ExportTrainBase {
  
  static ExportTrainBase createExporter(String expType,
                                        ForwardIndex fwdIndex,
                                        String queryExportFieldName, String indexExportFieldName,
                                        QrelReader qrelsTrain, QrelReader qrelsTest) {
    if (expType.compareToIgnoreCase(ExportTrainMatchZoo.FORMAT_NAME) == 0) {
      return new ExportTrainMatchZoo(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
    }
    if (expType.compareToIgnoreCase(ExportTrainCEDR.FORMAT_NAME) == 0) {
      return new ExportTrainCEDR(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
    }
    if (expType.compareToIgnoreCase(ExportTrainCEDRWithScores.FORMAT_NAME) == 0) {
      return new ExportTrainCEDRWithScores(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
    }
    return null;
  }

  static void addAllOptionDesc(Options opts) {
    ExportTrainMatchZoo.addOptionDesc(opts);
    ExportTrainNegSampleWithoutScoresBase.addOptionsDesc(opts);
    ExportTrainCEDR.addOptionsDesc(opts);
  }
  // Supposed to return an error message, if some options are missing or poorly formatted
  abstract String readAddOptions(CommandLine cmd);
  

  /**
   * 1. exportQuery function must be thread-safe: 
   *    make sure the SYNC output to the file.
   * 2. Each query can have more than one variant:
   *    i) The first variant are used only for candidate generation.
   *    It is likely be a stopped and lemmatized variant of the query without any punctuation.
   *    ii) The second variant can be less processed, e..g, it can retain punctuation.
   * 3. Test and or development queries can be processed quite differently.
   */ 
  
  /**
   * An abstract function for query export.
   * 
   * @param candProv        a candidate provider
   * @param queryNum        an ordinal query number
   * @param queryFields     a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}.
   * @param bIsTestQuery    true if the query is a test/dev query
   * @throws Exception
   */
  abstract void exportQuery(CandidateProvider candProv,
                            int queryNum, 
                            DataEntryFields queryFields,
                            boolean bIsTestQuery) throws Exception;
  
  
  abstract void startOutput() throws Exception;
  abstract void finishOutput() throws Exception;

  protected ExportTrainBase(ForwardIndex fwdIndex,
                           String queryExportFieldName, String indexExportFieldName, 
                           QrelReader qrelsTrain, QrelReader qrelsTest) {
    mFwdIndex = fwdIndex;
    mQueryExportFieldName = queryExportFieldName;
    mIndexExportFieldName = indexExportFieldName; 
    mQrelsTrain = qrelsTrain;
    mQrelsTest = qrelsTest;
  }

  protected String handleCase(String text) {
    if (text == null) {
      return null;
    }
    return mDoLowerCase ? text.toLowerCase() : text;
  }
  
  /**
   * Read and process document text if necessary. For raw indices, no processing is needed.
   *   
   * @param docId document ID
   * @return raw/parsed document text or null (if the document is not found or there is no 
   *         
   * @throws Exception
   */
  protected String getDocText(String docId) throws Exception {
    String text = null;
    if  (mFwdIndex.isTextRaw()) {
      text = mFwdIndex.getDocEntryTextRaw(docId);
    } else if (mFwdIndex.isParsed()) {
      text = mFwdIndex.getDocEntryParsedText(docId);
    } else {
      throw new RuntimeException("Export can be done only for text fields!");
    }
    if (text == null) {
      return null;
    }
    return handleCase(text.trim());
  }
  
  
  protected boolean mDoLowerCase = true;
 
  protected ForwardIndex              mFwdIndex;
  protected String                    mQueryExportFieldName;
  protected String                    mIndexExportFieldName;
  protected QrelReader                mQrelsTrain;
  protected QrelReader                mQrelsTest;


}
