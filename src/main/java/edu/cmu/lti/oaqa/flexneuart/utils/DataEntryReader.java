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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>An auto-closable resource class that reads possibly compressed entries in either JSONL or 
 * series-of-XML entries format. If, after removing the .gz or bz2. suffix,
 * the file has a .txt extension, it is assumed to be a series of XML entries.
 * If we move an optional .gz or .bz2 suffix and obtain a .jsonl suffix,
 * the input is assumed to be in a JSONL format.</p>
 * 
 * <p>All formats represent key-value pairs, where keys are always strings.
 * In XML values are only strings. In JSONL, values can be sometimes arrays of strings.
 * </p>
 * 
 * @author Leonid Boytsov
 *
 */
public class DataEntryReader implements java.lang.AutoCloseable {
  final static Logger logger = LoggerFactory.getLogger(DataEntryReader.class);
  
  private static final String QUERY_SUFFIX_BIN = ".bin";
  public static final String QUERY_SUFFIX_JSONL = ".jsonl";
  /**
   * 
   * @param fileName  a file name to check if the file is JSONL, or a binary format.
   *                  It will throw an exception if the extension is not jsonl, bin, or
   *                  jsonl with an additional gz/bzip2 suffix.
   * @return true if the file is in the JSONL format.
   * @throws IllegalArgumentException
   */
  public static boolean isFormatJSONL(String fileName) throws IllegalArgumentException {
    String fileNameNoCompr = CompressUtils.removeComprSuffix(fileName);
    if (fileNameNoCompr.endsWith(QUERY_SUFFIX_JSONL)) {
      return true;
    }
    if (fileNameNoCompr.endsWith(QUERY_SUFFIX_BIN)) {
      return false;
    }
    throw new IllegalArgumentException("Unexpected format (not jsonl or bin) in: " + fileName + 
                                      " after suffix removal: " + fileNameNoCompr);
  }

  /**
   * A constructor that guesses the data type from the file name suffix.
   * 
   * @param fileName  input file
   * @throws IllegalArgumentException
   * @throws IOException
   */
  public DataEntryReader(String fileName) throws IllegalArgumentException, IOException {
    mFileName = fileName;
    mIsJson = isFormatJSONL(mFileName);
    if (mIsJson) {
      mInpJson = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(mFileName))); 
      mInpBin = null;
    } else {
      mInpBin = new DataInputStream(CompressUtils.createInputStream(mFileName));
      mInpJson = null;
    }
  }
  
  /**
   * Read, parse, and validate the next entry. This function
   * returns an extended version of the data entry,
   * which can include fields with arrays of strings as values.
   * 
   * @return a null if we reached the end of the file, or (key, value) map that represents entry data.
   * @throws Exception 
   */
  public DataEntryFields readNext() throws Exception {
    mRecNum++;
    DataEntryFields currEntry = null;
    if (mIsJson) {
      String entryStr = JSONDataUtils.readNextJSONEntry(mInpJson, mRecNum);
      if (entryStr == null) {
        return null;
      }
      currEntry = JSONDataUtils.parseJSONEntry(entryStr, mRecNum);
    } else {      
      byte [] entryData = BSONDataUtils.readNextBSONEntry(mInpBin, mRecNum);
      if (entryData == null) {
        return null;
      }
      currEntry = BSONDataUtils.parseBSONEntry(entryData, mRecNum);
    }
    if (currEntry.mEntryId == null) {
      throw new RuntimeException("Missing entry ID entry: " + mRecNum );
    }
    return currEntry;
  }
  
  /**
   * Read and merge query data from JSONL and an optional binary input file.
   * It carries out sanity checks: if both files are present, they are 
   * expected to have the same number of entries with matching query IDs.
   * 
   * @param queryFilePrefix  query files prefix (without .json or .bin)
   * @return  an array of merged data entries
   * @throws Exception
   */
  public static ArrayList<DataEntryFields> readParallelQueryData(String queryFilePrefix) throws Exception {
    ArrayList<DataEntryFields>  res = new ArrayList<DataEntryFields>();
    HashSet<String> seen = new HashSet<String>();
    HashMap<String, DataEntryFields> entryIdToData = new HashMap<String, DataEntryFields>();
    
    // JSONL file should always be present
    String queryFileJSONL = queryFilePrefix + QUERY_SUFFIX_JSONL;
    try (DataEntryReader inp = new DataEntryReader(queryFileJSONL)) {
      DataEntryFields queryFields;
      for (int qnum = 1; (queryFields = inp.readNext()) != null; qnum++) {
        String qid = queryFields.mEntryId;
        if (qid == null) {
          logger.warn("Ignoring JSON query: # " + qnum + " because it has no query ID.");
          continue;
        }
        if (seen.contains(qid)) {
          throw new Exception("Repeating query ID , query: # " + qnum + " file: " + queryFileJSONL);
        }
        seen.add(qid);
        res.add(queryFields);
      }
    }
    
    Path queryFileBin = Paths.get(queryFilePrefix + QUERY_SUFFIX_BIN);
    if (Files.exists(queryFileBin)) {
      try (DataEntryReader inp = new DataEntryReader(queryFileBin.toString())) {
        DataEntryFields queryFields;
        for (int qnum = 1; (queryFields = inp.readNext()) != null; qnum++) {
          String qid = queryFields.mEntryId;
          if (qid == null) {
            logger.warn("Ignoring binary query: # " + qnum + " because it has no query ID.");
            continue;
          }
          if (entryIdToData.containsKey(qid)) {
            throw new Exception("Repeating query ID , query: # " + qnum + " file: " + queryFileBin);
          }
          entryIdToData.put(qid, queryFields);
        }
      }
      
      if (res.size() != entryIdToData.size()) {
        throw new Exception("JSONL and binary query files are out of sync, different number of entries. " +
                            "JSONL: " + res.size() + " binary: " + entryIdToData.size());
      }
      
      for (DataEntryFields queryFieldsJSONL : res) {
        String qid = queryFieldsJSONL.mEntryId;
        DataEntryFields queryEntryBIN = entryIdToData.get(qid);
        if (qid.compareTo(queryEntryBIN.mEntryId) != 0) {
          throw new RuntimeException("Bug: diverging query ids!");
        }
        queryFieldsJSONL.addAll(queryEntryBIN);
      }
    }
    
    return res;
  }
  
  @Override
  public void close() throws Exception {
    if (mInpJson != null) mInpJson.close();
    if (mInpBin != null) mInpBin.close();
    
  }  
  
  private final String mFileName;
  private int mRecNum = 0;
  private final BufferedReader mInpJson;
  private final DataInputStream mInpBin;
  private final boolean mIsJson;


}
