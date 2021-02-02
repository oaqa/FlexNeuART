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
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;


public abstract class ExportTrainBase {  
  static ExportTrainBase createExporter(String expType,
                                        ForwardIndex fwdIndex,
                                        QrelReader qrelsTrain, QrelReader qrelsTest) {
    if (expType.compareToIgnoreCase(ExportTrainMatchZoo.FORMAT_NAME) == 0) {
      return new ExportTrainMatchZoo(fwdIndex, qrelsTrain, qrelsTest);
    }
    if (expType.compareToIgnoreCase(ExportTrainCEDR.FORMAT_NAME) == 0) {
      return new ExportTrainCEDR(fwdIndex, qrelsTrain, qrelsTest);
    }
    return null;
  }

  static void addAllOptionDesc(Options opts) {
    ExportTrainMatchZoo.addOptionDesc(opts);
    ExportTrainNegSampleBase.addOptionsDesc(opts);
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
   * @param queryId         a string query ID
   * @param queryQueryText  a text of the query used for candidate generation
   * @param queryFieldText  a text for query that can be used to train models
   * @param bIsTestQuery    true if the query is a test/dev query
   * @throws Exception
   */
  abstract void exportQuery(CandidateProvider candProv,
                            int queryNum, 
                            String queryId,
                            String queryQueryText,
                            String queryFieldText,
                            boolean bIsTestQuery) throws Exception;
  
  
  abstract void startOutput() throws Exception;
  abstract void finishOutput() throws Exception;

  protected ExportTrainBase(ForwardIndex fwdIndex,
                           QrelReader qrelsTrain, QrelReader qrelsTest) {
    mFwdIndex = fwdIndex;
    mQrelsTrain = qrelsTrain;
    mQrelsTest = qrelsTest;
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
    if  (mFwdIndex.isRaw()) {
      return mFwdIndex.getDocEntryRaw(docId);
    }
    String text = mFwdIndex.getDocEntryParsedText(docId);
    if (text == null) {
      return null;
    }
    return text.trim();
  }
 
  protected ForwardIndex              mFwdIndex;
  protected QrelReader                mQrelsTrain;
  protected QrelReader                mQrelsTest;


}