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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.*;

/**
* One document entry of an in-memory forward index.
* 
* @author Leonid Boytsov
*
*/
class DocEntryExt implements Comparable<DocEntryExt> {
 public final DocEntryParsed  mDocEntry;
 public final String    mId;

 DocEntryExt(String id, DocEntryParsed docEntry) {
   mDocEntry = docEntry;
   mId = id;
 }

 @Override
 public int compareTo(DocEntryExt o) {
   return mId.compareTo(o.mId);
 }
}


/**
 * An in-memory forward index that is stored on disk in a simple text format.
 * This index needs to be loaded fully from disk before it can be used. 
 * This class is kept primarily for compatibility with old code, in particular,
 * to run unit tests. It *DOES NOT* support <b>raw</b> field and should not be normally used.
 * 
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexTextInMem extends ForwardIndex {  
  private static String NOT_SUPPORTED_PREFIX = ForwardIndexBackendType.inmem.toString() + " does not support";
 
  protected ForwardIndexTextInMem(String fileName) {
    mFileName = fileName;
  }

  @Override
  public void readIndex() throws Exception {
    String fileName = mFileName;
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
        
        final DocEntryParsed doc;
        
        try {
          doc = DocEntryParsed.fromString(docLine1 + '\n' + docLine2);
        } catch (Exception e) {
          throw new Exception(String.format("Error parsing entries in file %s lines: %d-%d, exception: %s", 
              fileName, lineNum-2, lineNum-1, e.toString()));
        }

        addDocEntryParsed(docId, doc);
        
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
    String fileName = mFileName;
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
  protected void postIndexCompAdd() {
    buildDocListSortedById();
  }

  @Override
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) {
  
  	DocEntryExt e = new DocEntryExt(docId, doc);
  	
		mStr2DocEntry.put(docId, doc);        
		mDocEntInAdditionOrder.add(e);
  }
  
  @Override
  public DocEntryParsed getDocEntryParsed(String docId) {
    return mStr2DocEntry.get(docId);
  }
  
  protected HashMap<String, DocEntryParsed> mStr2DocEntry = new HashMap<String, DocEntryParsed>();
  DocEntryExt[] mDocEntSortById = null;
  protected ArrayList<DocEntryExt> mDocEntInAdditionOrder = new ArrayList<DocEntryExt>();
  
  @Override
  protected void initIndex() throws IOException {
    // We do nothing here
  }

  // NOTE this function was never tested
  @Override
  public String[] getAllDocIds() {
    String res[] = new String[mDocEntSortById.length];
    
    for (int i = 0; i < mDocEntSortById.length; ++i) {
      res[i] = mDocEntSortById[i].mId; 
    }
    
    return res;
  }  

  private final String mFileName;

	@Override
	public String getDocEntryTextRaw(String docId) throws Exception {
		throw new RuntimeException(NOT_SUPPORTED_PREFIX + " raw text fields");
	}

	@Override
	protected void addDocEntryTextRaw(String docId, String docText) throws IOException {
	  throw new RuntimeException(NOT_SUPPORTED_PREFIX + " raw text fields");
	}

  @Override
  public byte[] getDocEntryBinary(String docId) throws Exception {
    throw new RuntimeException(NOT_SUPPORTED_PREFIX + " binary fields");
  }

  @Override
  protected void addDocEntryBinary(String docId, byte[] docBin) throws IOException {
    throw new RuntimeException(NOT_SUPPORTED_PREFIX + " binary fields");
  }
}
