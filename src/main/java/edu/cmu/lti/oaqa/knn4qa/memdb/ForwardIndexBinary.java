/**
 * 
 */
package edu.cmu.lti.oaqa.knn4qa.memdb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;

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

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;

/**
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinary extends ForwardIndex {
  
  public static final int COMMIT_INTERV = 500000;

  public ForwardIndexBinary(String filePrefix) throws IOException {
    this.mFilePrefix  = filePrefix;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds = new ArrayList<String>();
    
    String outputDirName = getBinaryDirName(mFilePrefix);
    File outputDir = new File(outputDirName);
    if (!outputDir.exists()) {
      if (!outputDir.mkdirs()) {
        System.out.println("couldn't create " + outputDir.getAbsolutePath());
        System.exit(1);
      }
    }
    if (!outputDir.isDirectory()) {
      System.out.println(outputDir.getAbsolutePath() + " is not a directory!");
      System.exit(1);
    }
    if (!outputDir.canWrite()) {
      System.out.println("Can't write to " + outputDir.getAbsolutePath());
      System.exit(1);
    }
    Analyzer analyzer = new WhitespaceAnalyzer();
    FSDirectory       indexDir    = FSDirectory.open(Paths.get(outputDirName));
    IndexWriterConfig indexConf   = new IndexWriterConfig(analyzer);
    
    /*
    OpenMode.CREATE creates a new index or overwrites an existing one.
    https://lucene.apache.org/core/6_0_0/core/org/apache/lucene/index/IndexWriterConfig.OpenMode.html#CREATE
    */
    indexConf.setOpenMode(OpenMode.CREATE); 
    indexConf.setRAMBufferSizeMB(LuceneCandidateProvider.RAM_BUFFER_SIZE);
    
    indexConf.setOpenMode(OpenMode.CREATE);
    mIndexWriter = new IndexWriter(indexDir, indexConf);  
  }
  
  @Override
  protected void sortDocEntries() {
    Collections.sort(mDocIds);
  }

  @Override
  public DocEntry getDocEntry(String docId) throws Exception {
    QueryParser parser = new QueryParser(UtilConst.TAG_DOCNO, mAnalyzer);
    Query       queryParsed = parser.parse(docId);
    
    TopDocs     hits = mSearcher.search(queryParsed, 1);
    ScoreDoc[]  scoreDocs = hits.scoreDocs;
    if (scoreDocs != null && scoreDocs.length == 1) {
      Document doc = mSearcher.doc(scoreDocs[0].doc);
      String docText = doc.get(UtilConst.TAG_DOC_ENTRY);
      return DocEntry.fromString(docText);
    }
    return null;
  }
  
  @Override
  public void readIndex() throws Exception {
    String fileName = mFilePrefix;
    BufferedReader  inp = null;
    
    try {
      inp = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

      String line;
      
      int lineNum = readHeader(fileName, inp);

      mDocIds = new ArrayList<String>();
      // Next read document entries
      lineNum++; line = inp.readLine();
     
      
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        String docId = line.trim();
        mDocIds.add(docId);
      }
      if (line == null) {
        throw new Exception(
            String.format(
                    "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                    lineNum, fileName));          
      }

      if (line != null) {
        if (!line.isEmpty()) {
          throw new Exception(
              String.format(
                      "Wrong format, expecting the end of flie at the number %d, file '%s'.",
                      lineNum, fileName));                  
        }

      }
      
      postIndexComp();
      
      String indexDirName = getBinaryDirName(mFilePrefix);
      
      mReader = DirectoryReader.open(FSDirectory.open(Paths.get(indexDirName)));
      mSearcher = new IndexSearcher(mReader);
      
      System.out.println("Finished loading context from file: " + fileName);
    } finally {    
      if (null != inp) inp.close();
    }
  }
  

  @Override
  protected void addDocEntry(String docId, DocEntry doc) throws IOException {   
    mDocIds.add(docId);
    Document luceneDoc = new Document();
    
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      System.out.println("Committing");
      mIndexWriter.commit();
    }
    
    luceneDoc.add(new StringField(UtilConst.TAG_DOCNO, docId, Field.Store.YES));
    luceneDoc.add(new StoredField(UtilConst.TAG_DOC_ENTRY, doc.toString()));
    mIndexWriter.addDocument(luceneDoc);
    
  }

  @Override
  public void saveIndex() throws IOException {
    String fileName = mFilePrefix;
    BufferedWriter out = null;
    
    try {
      out =    new BufferedWriter(
                  new OutputStreamWriter(
                      new FileOutputStream(fileName)));
      
      writeHeader(out);
      
      // 3. Write the document IDs
      for (String docId : mDocIds) {          
        out.write(docId);
        out.newLine();

      }
      out.newLine();
    } finally {
      if (out != null) out.close();
    }
   
    mIndexWriter.commit();
    mIndexWriter.close();
  }

  private ArrayList<String>   mDocIds = null;
  private IndexWriter mIndexWriter;
  private DirectoryReader mReader;
  private IndexSearcher mSearcher;
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();

  @Override
  public String[] getAllDocIds() {
    String res[] = new String[mDocIds.size()];
    
    for (int i = 0; i < mDocIds.size(); ++i) {
      res[i] = mDocIds.get(i);
    }
    
    return res;
  }
}
