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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;

/**
 * An auto-closable resource class that reads possibly compressed entries in either JSONL or 
 * series-of-XML entries format. If, after removing the .gz or bz2. suffix,
 * the file has a .txt extension, it is assumed to be a series of XML entries.
 * If we move an optional .gz or .bz2 suffix and obtain a .jsonl suffix,
 * the input is assumed to be in a JSONL format.
 * 
 * @author Leonid Boytsov
 *
 */
public class DataEntryReader implements java.lang.AutoCloseable {
  
  /**
   * 
   * @param fileName  a file name to check if the file is JSONL, or series-of-XML format.
   *                  It will throw an exception if the extension is not jsonl, txt, or
   *                  jsonl/txt with an additional gz/bzip2 suffix.
   * @return true if the file is in the JSONL format.
   * @throws IllegalArgumentException
   */
  public static boolean isFormatJSONL(String fileName) throws IllegalArgumentException {
    String fileNameNoCompr = CompressUtils.removeComprSuffix(fileName);
    if (fileNameNoCompr.endsWith(".jsonl")) {
      return true;
    }
    if (fileNameNoCompr.endsWith(".txt")) {
      return false;
    }
    throw new IllegalArgumentException("Unexpected extension (not jsonl or txt) in: " + fileName);
  }

  /**
   * A constructor that guesses the data type from the file name.
   * 
   * @param fileName  input file
   * @throws IllegalArgumentException
   * @throws IOException
   */
  public DataEntryReader(String fileName) throws IllegalArgumentException, IOException {
    mIsJson = isFormatJSONL(fileName);
    mInp = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fileName))); 
  }
  
  /**
   * Read, parse, and validate the next entry.
   * 
   * @return a null if we reached the end of the file, or (key, value) map that represents entry data.
   * @throws Exception 
   */
  public Map<String, String> readNext() throws Exception {
   
    if (mIsJson) {
      String doc = JSONUtils.readNextJSONEntry(mInp);
      if (doc == null) {
        return null;
      }
      return JSONUtils.parseJSONIndexEntry(doc);
    } else {
      String doc = XmlHelper.readNextXMLIndexEntry(mInp);
      if (doc == null) {
        return null;
      }
      return XmlHelper.parseXMLIndexEntry(doc);   
    }
  }
  
  @Override
  public void close() throws Exception {
    mInp.close();
  }  
  
  private final BufferedReader mInp;
  private final boolean mIsJson;


}