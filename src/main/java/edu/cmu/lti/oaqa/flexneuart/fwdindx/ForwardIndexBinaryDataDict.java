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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;

/**
 * A forward file that uses a key-value storage (a backend) directly. 
 * It compressed all raw text entries (GZIP when it reduces size and raw string, when it doesn't)
 * and converts regular {@link DocEntryParsed} entries to binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryDataDict extends ForwardIndexBinaryBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ForwardIndexBinaryDataDict.class);
 
  public static final int MEM_ALLOCATE_INCREMENT = 1024*1024*32; // Allocating memory in 32 MB chunks


  
  protected String mIndexPrefix = null;

  public ForwardIndexBinaryDataDict(String headerFile, PersistentKeyValBackend backend, String indexPrefix) {
    super(headerFile);
    mIndexPrefix = indexPrefix;
    mBackend = backend;
  }
  
  @Override
  protected void initIndex(int expectedQty) throws Exception {
    mBackend.initIndexForWriting(mIndexPrefix, expectedQty);
  }

  @Override
  public DocEntryParsed getDocEntryParsed(String docId) throws Exception {
    byte[] docBin =  mBackend.get(docId);
    if (docBin != null) {
      return DocEntryParsed.fromBinary(docBin);
    }
    return null;
  }  
  
  @Override
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) throws Exception {
  	byte binDoc[] = doc.toBinary();
  	mBackend.put(docId, binDoc);
  }
  
  @Override
  public String getDocEntryTextRaw(String docId) throws Exception {
    byte[] zippedStr =  mBackend.get(docId);
    if (zippedStr != null) {
      return CompressUtils.decomprStr(zippedStr);
    }
    return null;
  }

  @Override
  protected void addDocEntryTextRaw(String docId, String docText) throws Exception {
    mBackend.put(docId, CompressUtils.comprStr(docText));   
  }
  
  @Override
  public byte[] getDocEntryBinary(String docId) throws Exception {
    return mBackend.get(docId);
  }

  @Override
  protected void addDocEntryBinary(String docId, byte[] docBin) throws Exception {
    mBackend.put(docId, docBin); 
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeader();
    
    mBackend.openIndexForReading(mIndexPrefix);
    
    logger.info("Finished loading context from file: " + mIndexPrefix);
  }

  @Override
  public void saveIndex() throws Exception {
    writeHeader();
    mBackend.close();
  }

  @Override
  public String[] getAllDocIds() throws Exception {
    return mBackend.getKeyArray();
  }
  
  PersistentKeyValBackend  mBackend;

}
