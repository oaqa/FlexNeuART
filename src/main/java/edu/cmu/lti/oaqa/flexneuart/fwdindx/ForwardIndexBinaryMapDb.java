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

import java.io.IOException;
import java.util.concurrent.ConcurrentMap;

import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Serializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;

/**
 * A MapDB implementation of the forward file. It compressed all raw text entries
 * (GZIP when it reduces size and raw string, when it doesn't)
 * and converts regular {@link DocEntryParsed} entries to binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryMapDb extends ForwardIndexBinaryBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ForwardIndexBinaryMapDb.class);
 
  public static final int MEM_ALLOCATE_INCREMENT = 1024*1024*32; // Allocating memory in 32 MB chunks

  /* 
   * According to https://jankotek.gitbooks.io/mapdb/content/htreemap/
   * Maximal Hash Table Size is calculated as: segment# * node size ^ level count
   * So, the settings below give us approximately 134M : 16 * 16^5 
   * 
   */
  private static final int SEGMENT_QTY = 4;
  private static final int MAX_NODE_SIZE = 32;
  private static final int LEVEL_QTY = 5;
  
  protected String mBinFile = null;

  public ForwardIndexBinaryMapDb(String vocabAndDocIdsFile, String binFile) throws IOException {
    super(vocabAndDocIdsFile);
    mBinFile = binFile;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDb = DBMaker.fileDB(mBinFile)
                .allocateIncrement(MEM_ALLOCATE_INCREMENT)
                .closeOnJvmShutdown()
                .fileMmapEnable().make();
    // With respect to layout see https://jankotek.gitbooks.io/mapdb/content/htreemap/
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY)
                .layout(SEGMENT_QTY, MAX_NODE_SIZE, LEVEL_QTY)
                .counterEnable()
                .create();
  }

  @Override
  public DocEntryParsed getDocEntryParsed(String docId) throws Exception {
    byte[] docBin =  mDbMap.get(docId);
    if (docBin != null) {
      return DocEntryParsed.fromBinary(docBin);
    }
    return null;
  }  
  
  @Override
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) throws IOException {
  	byte binDoc[] = doc.toBinary();
    mDbMap.put(docId, binDoc);
  }
  
  @Override
  public String getDocEntryTextRaw(String docId) throws Exception {
    byte[] zippedStr =  mDbMap.get(docId);
    if (zippedStr != null) {
      return CompressUtils.decomprStr(zippedStr);
    }
    return null;
  }

  @Override
  protected void addDocEntryTextRaw(String docId, String docText) throws IOException {
    mDbMap.put(docId, CompressUtils.comprStr(docText));   
  }
  
  @Override
  public byte[] getDocEntryBinary(String docId) throws Exception {
    return mDbMap.get(docId);
  }

  @Override
  protected void addDocEntryBinary(String docId, byte[] docBin) throws IOException {
    mDbMap.put(docId, docBin); 
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeader();
    
    // Note that we disable file locking and concurrence to enable accessing the file by different programs at the same time
    mDb = DBMaker.fileDB(mBinFile)
                .allocateIncrement(MEM_ALLOCATE_INCREMENT)
                .concurrencyDisable()
                .fileLockDisable()
                .closeOnJvmShutdown()
                .fileMmapEnable()
                .readOnly()
                .make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY).open();
    
    logger.info("Finished loading context from file: " + mBinFile);
  }

  @Override
  public void saveIndex() throws IOException {
    writeHeader();
 
    mDb.commit();
    mDb.close();
  }

  @Override
  public String[] getAllDocIds() {
    String res[] = new String[0];
    return mDbMap.keySet().toArray(res);
  }
  
  private DB mDb;
  ConcurrentMap<String,byte[]> mDbMap;

}
