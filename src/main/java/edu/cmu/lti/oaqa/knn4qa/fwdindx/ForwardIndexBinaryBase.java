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
package edu.cmu.lti.oaqa.knn4qa.fwdindx;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;

/**
 * A base class for (mostly) binary forward indices. It can be possible
 * to have more than one. However, certain information such as the
 * list of IDs is stored for simplicity as plain text.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class ForwardIndexBinaryBase extends ForwardIndex {
  
  protected ForwardIndexBinaryBase(String vocabAndDocIdsFile) {
    mDocIds = new ArrayList<String>();
    mVocabAndDocIdsFile = vocabAndDocIdsFile;
  }
  
  /**
   * The function reads the header and the list of doc IDs. This is the 
   * information that is supposed to be stored in a text file of any binary
   * variant of the forward index.
   * 
   */
  protected void readHeaderAndDocIds() throws Exception {
    try (BufferedReader inp = new BufferedReader(new InputStreamReader(new FileInputStream(mVocabAndDocIdsFile)))) {

      String line;
      
      int lineNum = readHeader(mVocabAndDocIdsFile, inp);

      mDocIds.clear();
      // Next read document entries
      line = inp.readLine(); lineNum++; 
      
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        String docId = line.trim();
        mDocIds.add(docId);
      }
      if (line == null) {
        throw new Exception(
            String.format(
                    "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                    lineNum, mVocabAndDocIdsFile));          
      }

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
  
  protected void writeHeaderAndDocIds() throws IOException {
    try (BufferedWriter out =  new BufferedWriter(new OutputStreamWriter(new FileOutputStream(mVocabAndDocIdsFile)))) {
                                                
      writeHeader(out);
      
      // 3. Write the document IDs
      for (String docId : mDocIds) {          
        out.write(docId);
        out.newLine();

      }
      out.newLine();
    } 
  }
 
  @Override
  protected void sortDocEntries() {
    Collections.sort(mDocIds);
  }
  

  @Override
  public String[] getAllDocIds() {
    String res[] = new String[mDocIds.size()];
    
    for (int i = 0; i < mDocIds.size(); ++i) {
      res[i] = mDocIds.get(i);
    }
    
    return res;
  }
  
  protected final String              mVocabAndDocIdsFile;
  protected final ArrayList<String>   mDocIds;

}
