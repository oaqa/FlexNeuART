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
  
  public static final int COMMIT_INTERV = 1000000;
  public static final int MEM_ALLOCATE_INCREMENT = 1024*1024*256; // Allocating memory in 256 MB chunks
  
  protected String mBinFile = null;

  public ForwardIndexBinaryMapDb(String vocabAndDocIdsFile, String binFile) throws IOException {
    super(vocabAndDocIdsFile);
    mBinFile = binFile;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds.clear();
  
    mDb = DBMaker.fileDB(mBinFile).allocateIncrement(MEM_ALLOCATE_INCREMENT).closeOnJvmShutdown().fileMmapEnable().make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY).create();
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
  	mDocIds.add(docId);
  	byte binDoc[] = doc.toBinary();
    mDbMap.put(docId, binDoc);
    
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      System.out.println("Committing");
      mDb.commit();
    }
  }
  
  @Override
  public String getDocEntryRaw(String docId) throws Exception {
    byte[] zippedStr =  mDbMap.get(docId);
    if (zippedStr != null) {
      return CompressUtils.decomprStr(zippedStr);
    }
    return null;
  }

  @Override
  protected void addDocEntryRaw(String docId, String docText) throws IOException {
    mDocIds.add(docId);
    mDbMap.put(docId, CompressUtils.comprStr(docText));

    if (mDocIds.size() % COMMIT_INTERV == 0) {
      System.out.println("Committing");
      mDb.commit();
    }    
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeaderAndDocIds();
    
    // Note that we disable file locking and concurrence to enable accessing the file by different programs at the same time
    mDb = DBMaker.fileDB(mBinFile).allocateIncrement(MEM_ALLOCATE_INCREMENT).concurrencyDisable().fileLockDisable().closeOnJvmShutdown().fileMmapEnable().make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.BYTE_ARRAY).open();
    
    System.out.println("Finished loading context from file: " + mBinFile);
  }

  @Override
  public void saveIndex() throws IOException {
    writeHeaderAndDocIds();
   
    mDb.commit();
    mDb.close();
  }

  private DB mDb;
  ConcurrentMap<String,byte[]> mDbMap;
}
