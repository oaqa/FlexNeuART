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

import java.io.IOException;
import java.util.concurrent.ConcurrentMap;

import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Serializer;

/**
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryMapDb extends ForwardIndexBinaryBase {
  
  public static final int COMMIT_INTERV = 500000;
  
  protected String mBinFile = null;

  public ForwardIndexBinaryMapDb(String vocabAndDocIdsFile, String binFile) throws IOException {
    super(vocabAndDocIdsFile);
    mBinFile = binFile;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds.clear();
  
    mDb = DBMaker.fileDB(mBinFile).closeOnJvmShutdown().fileMmapEnable().make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.STRING).create();
  }


  @Override
  public DocEntry getDocEntry(String docId) throws Exception {
    String docText =  mDbMap.get(docId);
    if (docText != null) {
      return DocEntry.fromString(docText);
    }
    return null;
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeaderAndDocIds();
    
    mDb = DBMaker.fileDB(mBinFile).closeOnJvmShutdown().fileMmapEnable().make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.STRING).open();
    
    System.out.println("Finished loading context from file: " + mBinFile);
  }
  

  @Override
  protected void addDocEntry(String docId, DocEntry doc) throws IOException {   
    mDocIds.add(docId);
    mDbMap.put(docId, doc.toString());
    
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      System.out.println("Committing");
      mDb.commit();
    }

  }

  @Override
  public void saveIndex() throws IOException {
    writeHeaderAndDocIds();
   
    mDb.commit();
    mDb.close();
  }

  private DB mDb;
  ConcurrentMap<String,String> mDbMap;
}
