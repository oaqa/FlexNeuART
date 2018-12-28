/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.memdb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.*;

class WordEntryExt implements Comparable<WordEntryExt> {
  WordEntry mWordEntry;
  String    mWord;

  WordEntryExt(String word, WordEntry wordEntry) {
    mWordEntry = wordEntry;
    mWord = word;
  }
  @Override
  public int compareTo(WordEntryExt o) {
    return mWordEntry.mWordId - o.mWordEntry.mWordId;
  }
}

/**
 * 
 * @author Leonid Boytsov
 *
 */
public class InMemForwardIndexText extends ForwardIndex {  
 
  protected InMemForwardIndexText(String filePrefix) {
    mFilePrefix = filePrefix;
  }

  @Override
  public void readIndex() throws Exception {
    String fileName = mFilePrefix;
    BufferedReader  inp = null;
    
    try {
      inp = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

      String line;
      
      int lineNum = readHeader(fileName, inp);

      // Next read document entries
      lineNum++; line = inp.readLine();
      
      String docLine1, docLine2;
      
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        String docId = line.trim();
        lineNum++; docLine1 = inp.readLine();
        if (docLine1 == null) {
          throw new Exception(
              String.format(
                      "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                      lineNum, fileName));          
        }
        lineNum++; docLine2 = inp.readLine();
        if (docLine2 == null) {
          throw new Exception(
              String.format(
                      "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                      lineNum, fileName));          
        }
        
        final DocEntry doc;
        
        try {
          doc = DocEntry.fromString(docLine1 + '\n' + docLine2);
        } catch (Exception e) {
          throw new Exception(String.format("Error parsing entries in file %s lines: %d-%d, exception: %s", 
              fileName, lineNum-2, lineNum-1, e.toString()));
        }

        addDocEntry(docId, doc);
        
      }
      if (line == null) {
        throw new Exception(
            String.format(
                    "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                    lineNum, fileName));          
      }
      line = inp.readLine(); ++lineNum;
      if (line != null) {
        if (!line.isEmpty()) {
          throw new Exception(
              String.format(
                      "Wrong format, expecting the end of flie at the number %d, file '%s'.",
                      lineNum, fileName));                  
        }

      }
      
      postIndexComp();
      
      System.out.println("Finished loading context from file: " + fileName);
    } finally {    
      if (null != inp) inp.close();
    }
  }
  
  @Override
  public void saveIndex() throws IOException {
    String fileName = mFilePrefix;
    BufferedWriter out = null;
    
    try {
      out =    new BufferedWriter(
                  new OutputStreamWriter(
                      new FileOutputStream(fileName)));
      
      writeHeader(out);
      
      // 3. Write the document entries
      for (DocEntryExt e : mDocEntSortById) {
        
        out.write(e.mId);
        out.newLine();
        
        out.write(e.mDocEntry.toString());
        /* 
         * We add a new line, b/c mDocEntry.toString() doesn't generate a new line.
         * However, we do not want an empty separating line between document entries.
         */
        out.newLine(); 
      }
      out.newLine();
    } finally {
      if (out != null) out.close();
    }
  }
  
  void buildDocListSortedById() {
    mDocEntSortById = new DocEntryExt[mDocEntInAdditionOrder.size()];
    int k = 0;
    for (DocEntryExt e : mDocEntInAdditionOrder) {
      mDocEntSortById[k++] = e;
    }
    Arrays.sort(mDocEntSortById);
  }
  
  @Override
  protected void sortDocEntries() {
    buildDocListSortedById();
  }

  @Override
  protected void addDocEntry(String docId, DocEntry doc) {
  
    mStr2DocEntry.put(docId, doc);        
    mDocEntInAdditionOrder.add(new DocEntryExt(docId, doc)); 
  }
  
  @Override
  public DocEntry getDocEntry(String docId) {
    return mStr2DocEntry.get(docId);
  }
  
  protected HashMap<String, DocEntry> mStr2DocEntry = new HashMap<String, DocEntry>();
  DocEntryExt[] mDocEntSortById = null;
  protected ArrayList<DocEntryExt> mDocEntInAdditionOrder = new ArrayList<DocEntryExt>();
  
  @Override
  protected void initIndex() throws IOException {
    // We do nothing here
  }  


}
