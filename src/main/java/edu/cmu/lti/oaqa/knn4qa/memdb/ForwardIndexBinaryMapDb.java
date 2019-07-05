/**
 * 
 */
package edu.cmu.lti.oaqa.knn4qa.memdb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ConcurrentMap;


import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Serializer;

/**
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryMapDb extends ForwardIndex {
  
  public static final int COMMIT_INTERV = 500000;

  public ForwardIndexBinaryMapDb(String filePrefix) throws IOException {
    this.mFilePrefix  = filePrefix;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds = new ArrayList<String>();
    
    String outputFileName = getBinaryDirOrFileName(mFilePrefix);

    mDb = DBMaker.fileDB(outputFileName).closeOnJvmShutdown().fileMmapEnable().make();
    mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.STRING).create();
  }
  
  @Override
  protected void sortDocEntries() {
    Collections.sort(mDocIds);
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
      
      String indexFileName = getBinaryDirOrFileName(mFilePrefix);
      
      mDb = DBMaker.fileDB(indexFileName).closeOnJvmShutdown().fileMmapEnable().make();
      mDbMap = mDb.hashMap("map", Serializer.STRING, Serializer.STRING).open();
      
      System.out.println("Finished loading context from file: " + fileName);
    } finally {    
      if (null != inp) inp.close();
    }
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
   
    mDb.commit();
    mDb.close();
  }

  private ArrayList<String>   mDocIds = null;
  private DB mDb;
  ConcurrentMap<String,String> mDbMap;

  @Override
  public String[] getAllDocIds() {
    String res[] = new String[mDocIds.size()];
    
    for (int i = 0; i < mDocIds.size(); ++i) {
      res[i] = mDocIds.get(i);
    }
    
    return res;
  }
}
