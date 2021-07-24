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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

/**
 * A base class for (mostly) binary forward indices, which keeps
 * meta information in text format (words stat and doc ids). 
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class ForwardIndexBinaryBase extends ForwardIndex {
  
  protected ForwardIndexBinaryBase(String vocabAndDocIdsFile) {
    mVocabAndDocIdsFile = vocabAndDocIdsFile;
  }
  
  /**
   * The function reads the header and the list of doc IDs. This is the 
   * information that is supposed to be stored in a text file of any binary
   * variant of the forward index.
   * 
   */
  protected void readHeader() throws Exception {
  	
    try (BufferedReader inp = new BufferedReader(new InputStreamReader(new FileInputStream(mVocabAndDocIdsFile)))) {

      String line;
      
      int lineNum = readHeader(mVocabAndDocIdsFile, inp);

      line = inp.readLine(); lineNum++; 

      if (line != null) {
        if (!line.isEmpty()) {
          throw new Exception(
              String.format(
                      "Wrong format, expecting the end of flie at the number %d, file '%s'.",
                      lineNum, mVocabAndDocIdsFile));                  
        }

      }
      
      postIndexComp();      
    }
  }
  
  protected void writeHeader() throws IOException {
    try (BufferedWriter out =  new BufferedWriter(new OutputStreamWriter(new FileOutputStream(mVocabAndDocIdsFile)))) {                                               
      writeHeader(out);
    } 
  }
 
  @Override
  protected void postIndexCompAdd() {
    // Potentially can do some useful extra work after the index is loaded
  }
  
  protected final String              mVocabAndDocIdsFile;
}
