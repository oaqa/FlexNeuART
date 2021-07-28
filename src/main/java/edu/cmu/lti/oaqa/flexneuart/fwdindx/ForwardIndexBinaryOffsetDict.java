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

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.Iterator;

import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class QtyOffsetPair {
	public final int mQty;
	public final long mOffset;
	
	public QtyOffsetPair(int qty, long off) {
		this.mQty = qty;
		this.mOffset = off;
	}

	@Override
	public String toString() {
		return "QtyOffsetPair [mQty=" + mQty + ", mOffset=" + mOffset + "]";
	}
}

/**
 * A forward file that uses a key-value storage (a backend) to store the offsets to data entries,
 * which are, in turn, are stored in a separate file. 
 * It compressed all raw text entries (GZIP when it reduces size and raw string, when it doesn't)
 * and converts regular {@link DocEntryParsed} entries to binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryOffsetDict extends ForwardIndexBinaryBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ForwardIndexBinaryOffsetDict.class);
  public static final String DATA_SUFFIX = ".dat";
  
  protected String mIndexPrefix = null;

  public ForwardIndexBinaryOffsetDict(String headerFile, PersistentKeyValBackend backend, String indexPrefix)  {
    super(headerFile);
    mIndexPrefix = indexPrefix;
    mBackend = backend;
  }
  
  @Override
  protected void initIndex(int expectedQty) throws Exception {
    mDocQty = 0;
    mDataFileWrite = new BufferedOutputStream(new FileOutputStream(new File(mIndexPrefix + DATA_SUFFIX)));
    
    /*
    File outputDir = new File(mIndexPrefix);
    if (!outputDir.exists()) {
      if (!outputDir.mkdirs()) {
        logger.error("couldn't create " + outputDir.getAbsolutePath());
        System.exit(1);
      }
    }
    if (!outputDir.isDirectory()) {
      logger.error(outputDir.getAbsolutePath() + " is not a directory!");
      System.exit(1);
    }
    if (!outputDir.canWrite()) {
      logger.error("Can't write to " + outputDir.getAbsolutePath());
      System.exit(1);
    }
    */
    
    mBackend.initIndexForWriting(mIndexPrefix, expectedQty);
  }

  @Override
  public DocEntryParsed getDocEntryParsed(String docId) throws Exception {
    byte[] docBin =  this.getDocEntryPacked(docId);
    if (docBin != null) {
      return DocEntryParsed.fromBinary(docBin);
    }
    return null;
  }

  @Override
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) throws Exception {
  	addDocEntryPacked(docId, doc.toBinary());        
  }
  
	@Override
	public String getDocEntryTextRaw(String docId) throws Exception {
    byte[] zippedStr = this.getDocEntryPacked(docId);
    if (zippedStr != null) {
      return CompressUtils.decomprStr(zippedStr);
    }
    return null;
	}

	@Override
	protected void addDocEntryTextRaw(String docId, String docText) throws Exception {
		addDocEntryPacked(docId, CompressUtils.comprStr(docText));
  }
	
  @Override
  public byte[] getDocEntryBinary(String docId) throws Exception {
    return getDocEntryPacked(docId);
  }

  @Override
  protected void addDocEntryBinary(String docId, byte[] docBin) throws Exception {
    addDocEntryPacked(docId, docBin);
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeader();
    
    mBackend.openIndexForReading(mIndexPrefix);
    
    mDataFileRead = new RandomAccessFile(mIndexPrefix + DATA_SUFFIX, "r");
    
    logger.info("Finished loading context from: " + mIndexPrefix);
  }

  @Override
  public void saveIndex() throws Exception {
    writeHeader();
   
    mDataFileWrite.close();
    mBackend.close();
  }
  
  private void addDocEntryPacked(String docId, byte [] binEntry)  throws Exception {  	
    byte [] offLenPair = intLongPairToBytes(binEntry.length, mDataSize);
    
    mBackend.put(docId, offLenPair);
    
    mDataFileWrite.write(binEntry);
    mDataSize += binEntry.length;
  }
    
  private byte [] getDocEntryPacked(String docId) throws Exception {
    byte[] bdata = mBackend.get(docId);
  	
  	QtyOffsetPair qtyOff = bytesToIntLongPair(bdata);
  	
  	byte res [] = new byte[qtyOff.mQty];
  	
  	// random access files are not thread safe!
  	synchronized (this) {
		  mDataFileRead.seek(qtyOff.mOffset);
		  int readQty = mDataFileRead.read(res, 0, qtyOff.mQty);
		  if (readQty != qtyOff.mQty) {
		  	throw new IOException("Read only " + readQty + " bytes out of " + qtyOff.mQty);
		  }
  	}
  	  	
  	return res;
  }
  
  public static byte[] intLongPairToBytes(int qty, long off) {
    ByteBuffer out = ByteBuffer.allocate(4 + 8);
    out.order(Const.BYTE_ORDER);
    out.putInt(qty);
    out.putLong(off);
    // The array should be fully filled up
    return out.array();  	
  }
  
  public static QtyOffsetPair bytesToIntLongPair(byte[] buf) {
  	ByteBuffer in = ByteBuffer.wrap(buf);
  	in.order(Const.BYTE_ORDER);  	  	
  	
  	return new QtyOffsetPair(in.getInt(), in.getLong());
  }
  

  @Override
  public String[] getAllDocIds() throws Exception {
    return mBackend.getKeyArray();
  }
  
  @Override
  public Iterator<String> getDocIdIterator() throws Exception {
    return mBackend.getKeyIterator();
  }

  RandomAccessFile 								mDataFileRead;
  BufferedOutputStream            mDataFileWrite;
  PersistentKeyValBackend         mBackend;

  long          mDataSize = 0;

}
