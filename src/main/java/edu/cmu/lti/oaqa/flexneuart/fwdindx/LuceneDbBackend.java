package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.nio.file.Paths;

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
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;

public class LuceneDbBackend extends PersistentKeyValBackend {
  private static final Logger logger = LoggerFactory.getLogger(PersistentKeyValBackend.class);
  
  public static final String BIN_DATA_FIELD_NAME = "bin_data";

  @Override
  public void openIndexForReading(String indexPrefix) throws Exception {
    mReader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPrefix)));
    mSearcher = new IndexSearcher(mReader);
    
    logger.info("LuceneDB opened for reading: " + indexPrefix);
  }

  @Override
  public void initIndexForWriting(String indexPrefix, int expectedQty) throws Exception {
    FSDirectory       indexDir    = FSDirectory.open(Paths.get(indexPrefix));
    IndexWriterConfig indexConf   = new IndexWriterConfig(mAnalyzer);
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
  public void put(String key, byte[] value) throws Exception  {
    Document luceneDoc = new Document();
    
    luceneDoc.add(new StringField(Const.DOC_ID_FIELD_NAME, key, Field.Store.YES));
    luceneDoc.add(new StoredField(BIN_DATA_FIELD_NAME, new BytesRef(value)));

    // Index writers should be completely thread-safe 
    mIndexWriter.addDocument(luceneDoc);
  }

  @Override
  public void close() throws Exception {
    if (mIndexWriter != null) {
      mIndexWriter.commit();
      mIndexWriter.close();
      mIndexWriter = null;
    }
    if (mReader != null) {
      mReader.close();
      mReader = null;
    }
    mSearcher = null;
  }

  @Override
  public byte[] get(String key) throws Exception {
    QueryParser parser = new QueryParser(Const.DOC_ID_FIELD_NAME, mAnalyzer);
    Query       queryParsed = parser.parse(key);
    
    TopDocs     hits = mSearcher.search(queryParsed, 1);
    ScoreDoc[]  scoreDocs = hits.scoreDocs;

    if (scoreDocs != null && scoreDocs.length == 1) {
      Document doc = mSearcher.doc(scoreDocs[0].doc);
      return doc.getBinaryValue(BIN_DATA_FIELD_NAME).bytes;
    } else {
      return null;
    }
  }

  @Override
  public String[] getKeyArray() throws Exception {
    int qty = size();
    String res[] = new String[qty];
    
    for (int i = 0; i < qty; ++i) {
      res[i] = mSearcher.doc(i).get(Const.DOC_ID_FIELD_NAME);
    }
    
    return res;
  }

  @Override
  public int size() throws Exception {
    if (mReader == null) {
      throw new Exception("Trying to get a # of documents on an index that is not open!");
    }
    return mReader.getDocCount(BIN_DATA_FIELD_NAME);
  }
  
  private IndexWriter mIndexWriter;
  private DirectoryReader mReader;
  private IndexSearcher mSearcher;
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();

}
