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
import java.nio.file.Paths;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
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
 * A hybrid implementation of the binary forward file where the
 * data entries  
 * It compressed all raw text entries
 * (GZIP when it reduces size and raw string, when it doesn't)
 * and converts regular {@link DocEntryParsed} entries to binary format.
 * 
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryFlatFileData extends ForwardIndexBinaryBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ForwardIndexBinaryFlatFileData.class);
  public static final int COMMIT_INTERV = 2000000;
  public static final String DATA_SUFFIX = ".dat";
  
  protected String mBinFile = null;

  public ForwardIndexBinaryFlatFileData(String vocabAndDocIdsFile, String binFile) throws IOException {
    super(vocabAndDocIdsFile);
    mBinFile = binFile;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds.clear();
  
    mDataFileWrite = new BufferedOutputStream(new FileOutputStream(new File(mBinFile + DATA_SUFFIX)));
    
    File outputDir = new File(mBinFile);
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
    Analyzer analyzer = new WhitespaceAnalyzer();
    FSDirectory       indexDir    = FSDirectory.open(Paths.get(mBinFile));
    IndexWriterConfig indexConf   = new IndexWriterConfig(analyzer);
    
    /*
    OpenMode.CREATE creates a new index or overwrites an existing one.
    https://lucene.apache.org/core/6_0_0/core/org/apache/lucene/index/IndexWriterConfig.OpenMode.html#CREATE
    */
    indexConf.setOpenMode(OpenMode.CREATE); 
    indexConf.setRAMBufferSizeMB(LuceneCandidateProvider.DEFAULT_RAM_BUFFER_SIZE);
    
    indexConf.setOpenMode(OpenMode.CREATE);
    mIndexWriter = new IndexWriter(indexDir, indexConf);  
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
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) throws IOException {
  	addDocEntryPacked(docId, doc.toBinary());        
  }
  
	@Override
	public String getDocEntryRaw(String docId) throws Exception {
    byte[] zippedStr = this.getDocEntryPacked(docId);
    if (zippedStr != null) {
      return CompressUtils.decomprStr(zippedStr);
    }
    return null;
	}

	@Override
	protected void addDocEntryRaw(String docId, String docText) throws IOException {
		addDocEntryPacked(docId, CompressUtils.comprStr(docText));
   }
  
  @Override
  public void readIndex() throws Exception {
    readHeaderAndDocIds();
    
    mReader = DirectoryReader.open(FSDirectory.open(Paths.get(mBinFile)));
    mSearcher = new IndexSearcher(mReader);
    
    mDataFileRead = new RandomAccessFile(mBinFile + DATA_SUFFIX, "r");
    
    logger.info("Finished loading context from: " + mBinFile);
  }

  @Override
  public void saveIndex() throws IOException {
    writeHeaderAndDocIds();
   
    mDataFileWrite.close();
    
    mIndexWriter.commit();
    mIndexWriter.close();
  }
  
  private void addDocEntryPacked(String docId, byte [] binEntry)  throws IOException {  	
		mDocIds.add(docId);
    
    byte [] offLenPair = intLongPairToBytes(binEntry.length, mDataSize);
    
    Document luceneDoc = new Document();
    
    luceneDoc.add(new StringField(Const.TAG_DOCNO, docId, Field.Store.YES));
    luceneDoc.add(new StoredField(Const.TAG_DOC_ENTRY, new BytesRef(offLenPair)));

    // Index writers should be completely thread-safe 
    mIndexWriter.addDocument(luceneDoc);
    
    mDataFileWrite.write(binEntry);
    mDataSize += binEntry.length;
  			
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      logger.info("Committing");
      mIndexWriter.commit();
    }
  }
    
  private byte [] getDocEntryPacked(String docId) throws IOException, ParseException {
    QueryParser parser = new QueryParser(Const.TAG_DOCNO, mAnalyzer);
    Query       queryParsed = parser.parse(docId);
    
    TopDocs     hits = mSearcher.search(queryParsed, 1);
    ScoreDoc[]  scoreDocs = hits.scoreDocs;

    byte bdata[] = null;
    if (scoreDocs != null && scoreDocs.length == 1) {
      Document doc = mSearcher.doc(scoreDocs[0].doc);
      bdata = doc.getBinaryValue(Const.TAG_DOC_ENTRY).bytes;
    } else {
    	return null;
    }
  	
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

  RandomAccessFile 								mDataFileRead;
  BufferedOutputStream            mDataFileWrite;
  
  private IndexWriter mIndexWriter;
  private DirectoryReader mReader;
  private IndexSearcher mSearcher;
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();
  
  long                            mDataSize = 0;
}
