/*
 *  Copyright 2015 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.memdb;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.*;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;

class WordEntryExt implements Comparable<WordEntryExt> {
  WordEntry mWordEntry;
  String    mWord;

  WordEntryExt(String word, WordEntry wordEntry) {
    mWordEntry = wordEntry;
    mWord = word;
  }
  @Override
  public int compareTo(WordEntryExt o) {
    return mWordEntry.mWordId - o.mWordEntry.mWordId;
  }
}

/**
 * 
 * An in-memory forward index. 
 * 
 * <p>This simple class can read XML files produced by a pipeline
 * and collect all unique (space-separated) words that appear in all the fields.
 * These are used to (1) create an in-memory dictionary (2) an in-memory forward index.
 * The in-memory forward index can be stored to and loaded from the file. </p>
 * 
 * <p><b>NOTE:</b> word IDs start from 1.</p>
 * 
 * <p>
 * In addition, it computes some statistics for each field:
 * </p>
 * <ol> 
 *  <li>An average number of words per document;
 *  <li>The total number of documents;
 *  <li>An number of documents where each word occur;
 * </ol>  
 * 
 * @author Leonid Boytsov
 *
 */
public class InMemForwardIndex {  
  public static final String DOCLEN_QTY_PREFIX = "@ ";
  public static final WordEntry UNKNOWN_WORD = new WordEntry(-1);
  public static final int MIN_WORD_ID = 1;
  
  /**
   * Constructor: Creates an index from one or more files (for a given field name).
   * 
   * @param fieldName         the name of the field (as specified in the SOLR index-file)
   * @param fileNames         an array of files from which the index is created
   * @param bStoreWordIdSeq   if true, we memorize the sequence of word IDs, otherwise only a number of words (doc. len.)
   * @param maxNumRec         the maximum number of records to process
   * @throws IOException
   */
  public InMemForwardIndex(String fieldName, String[] fileNames, 
                          boolean bStoreWordIdSeq, 
                          int maxNumRec) throws IOException {    
    mDocQty       = 0;
    mTotalWordQty = 0;
    
    int totalUniqWordQty = 0; // sum the number of uniq words per document (over all documents)
    
    System.out.println("Creating a new in-memory forward index, maximum # of docs to process: " + maxNumRec);
    
    for (String fileName : fileNames) {    
      BufferedReader  inpText = new BufferedReader(
          new InputStreamReader(CompressUtils.createInputStream(fileName)));
      
      String docText = XmlHelper.readNextXMLIndexEntry(inpText);
  
      for (;mDocQty < maxNumRec && docText!= null; 
           docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
        Map<String, String>         docFields = null;
  
        try {
          docFields = XmlHelper.parseXMLIndexEntry(docText);
        } catch (Exception e) {
          System.err.println(String.format("Parsing error, offending DOC #%d:\n%s", mDocQty, docText));
          System.exit(1);
        }
        
        String docId = docFields.get(UtilConst.TAG_DOCNO);
        
        if (docId == null) {
          System.err.println(String.format("No ID tag '%s', offending DOC #%d:\n%s", 
                                            UtilConst.TAG_DOCNO, mDocQty, docText));
        }
        
        String text = docFields.get(fieldName);
        if (text == null) text = "";
        if (text.isEmpty()) {
          System.out.println(String.format("Warning: empty field '%s' for document '%s'",
                                           fieldName, docId));
        }
        
        // If the string is empty, the array will contain an emtpy string, but
        // we don't want this
        text=text.trim();
        String words[] = text.isEmpty() ? new String[0] : text.split("\\s+");
  
        // First obtain word IDs for unknown words
        for (int i = 0; i < words.length; ++i) {
          String w = words[i];
          WordEntry wEntry = mStr2WordEntry.get(w);
          if (null == wEntry) {
            wEntry = new WordEntry(MIN_WORD_ID + mStr2WordEntry.size());
            mStr2WordEntry.put(w, wEntry);
          }
        }
        
        DocEntry doc = createDocEntry(words, bStoreWordIdSeq);
        
        mStr2DocEntry.put(docId, doc);        
        mDocEntInAdditionOrder.add(new DocEntryExt(docId, doc)); 
        
        HashSet<String> uniqueWords = new HashSet<String>();        
        for (String w: words) uniqueWords.add(w);
        
        // Let's update word co-occurrence statistics
        for (String w: uniqueWords) {
          WordEntry wEntry = mStr2WordEntry.get(w);
          wEntry.mWordFreq++;
        }
        
        ++mDocQty;
        mTotalWordQty += words.length;
        totalUniqWordQty += doc.mQtys.length;
      }
      
      postIndexComp();
      
      System.out.println("Finished processing file: " + fileName);
      
      if (mDocQty >= maxNumRec) break;
    }
    
    System.out.println("Final statistics: ");
    System.out.println(
        String.format("Number of documents %d, total number of words %d, average reduction due to keeping only unique words %f",
                      mDocQty, mTotalWordQty, 
                      ((double)mTotalWordQty)/totalUniqWordQty));
  }
  
  /**
   * Constructor: retrieves a previously stored index.
   * 
   * @param fileName the file generated by the function {@link #save(String)}.
   */
  public InMemForwardIndex(String fileName) throws Exception {
    BufferedReader  inp = null;
    
    try {
      inp = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

      String meta = inp.readLine();

      if (null == meta)
        throw new Exception(
            String.format(
                    "Can't read meta information: the file '%s' may have been truncated.",
                    fileName));

      String parts[] = meta.split("\\s+");
      if (parts.length != 2)
        throw new Exception(
            String.format(
                    "Wrong format, file '%s': meta-information (first line) should contain exactly two space-separated numbers.",
                    fileName));

      try {
        mDocQty = Integer.parseInt(parts[0]);
        mTotalWordQty = Long.parseLong(parts[1]);
      } catch (NumberFormatException e) {
        throw new Exception(String.format(
            "Invalid meta information (should be two-integers), file '%s'.",
            fileName));
      }

      String line = inp.readLine();
      if (line == null || !line.isEmpty()) {
        String.format(
                "Can't read an empty line after meta information: the file '%s' may have been truncated.",
                fileName);
      }

      // First read the dictionary
      int lineNum = 3;
      line = inp.readLine();
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        parts = line.split("\\t");
        if (parts.length != 2) {
          throw new Exception(
              String.format(
                      "Invalid dictionary format (should be two tab-separated parts), line %d, file %s",
                      lineNum, fileName));
        }
        String w = parts[0];
        String[] partSuff = parts[1].split(":");
        if (partSuff.length != 2) {
          throw new Exception(
              String.format(
                      "Invalid dictionary entry format (should end with two colon separated integers), line %d, file %s",
                      lineNum, fileName));
        }

        int wordId = -1;
        int docQty = -1;

        try {
          wordId = Integer.parseInt(partSuff[0]);
          docQty = Integer.parseInt(partSuff[1]);
        } catch (NumberFormatException e) {
          throw new Exception(
              String.format(
                      "Invalid dictionary entry format (an ID or count isn't integer), line %d, file %s",
                      lineNum, fileName));
        }
        if (wordId < MIN_WORD_ID) {
          throw new Exception(
                      String.format("Inconsistent data, wordId %d is too small, should be>= %d", 
                                    wordId, MIN_WORD_ID));
        }
        mStr2WordEntry.put(w, new WordEntry(wordId, docQty));
      }
      if (line == null)
        throw new Exception(
            String.format(
                    "Can't read an empty line (line number %d): the file '%s' may have been truncated.",
                    lineNum, fileName));

      // Next read document entries
      lineNum++; line = inp.readLine();
      
      for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
        String docId = line.trim();
        lineNum++; line = inp.readLine();
        if (line == null) {
          throw new Exception(
              String.format(
                      "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                      lineNum, fileName));          
        }

        ArrayList<Integer> wordIds = new ArrayList<Integer>(), wordQtys = new ArrayList<Integer>();
        
        if (!line.isEmpty()) {
          parts = line.split("\\s+");
          int k = 0;
          for (String s: parts) {
            String [] parts2 = s.split(":");
            ++k;
            if (parts2.length != 2) {
              throw new Exception(
                  String.format(
                      "Wrong format (line number %d), file '%s': expecting two colon-separated numbers in the substring:" +
                      "'%s' (part # %d)",
                      lineNum, fileName, s, k));          
              
            }
            
            try {
              int wordId = Integer.parseInt(parts2[0]);
              int wordQty = Integer.parseInt(parts2[1]);
              
              wordIds.add(wordId);
              wordQtys.add(wordQty);
            } catch (NumberFormatException e) {
              throw new Exception(
                  String.format(
                          "Invalid document entry format (an ID or count isn't integer), line %d, file %s in the substring '%s' (part #%d)",
                          lineNum, fileName, s, k));
            }          
          }
        }
        lineNum++; line = inp.readLine();
        if (line == null) {
          throw new Exception(
              String.format(
                      "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                      lineNum, fileName));          
        }
        
        
        
        final DocEntry doc;
        
        if (line.startsWith(DOCLEN_QTY_PREFIX)) {
          final int docLen;
          try {
            docLen = Integer.parseInt(line.substring(DOCLEN_QTY_PREFIX.length()));
          } catch (NumberFormatException e) {
            throw new Exception(
                String.format(
                        "Document length isn't integer (line number %d, file '%s')",
                        lineNum, fileName));               
          }
          doc = new DocEntry(wordIds, wordQtys, docLen); 
        } else {
          ArrayList<Integer> wordIdSeq = new ArrayList<Integer>();
          
          if (!line.isEmpty()) {
            parts = line.split("\\s+");
            int k = 0;
            for (String s: parts) {
              ++k;
              try {
                int wordId = Integer.parseInt(s);
                wordIdSeq.add(wordId);
              } catch (NumberFormatException e) {
                throw new Exception(
                    String.format(
                            "Invalid document entry format (word ID isn't integer), line %d, file %s in the substring '%s' (part #%d)",
                            lineNum, fileName, s, k));
              }          
            }
          }
          doc = new DocEntry(wordIds, wordQtys, wordIdSeq, true);
        }
        

        mStr2DocEntry.put(docId, doc);
        mDocEntInAdditionOrder.add(new DocEntryExt(docId, doc));
      }
      if (line == null) {
        throw new Exception(
            String.format(
                    "Can't read a document line (line number %d): the file '%s' may have been truncated.",
                    lineNum, fileName));          
      }
      line = inp.readLine(); ++lineNum;
      if (line != null) {
        if (!line.isEmpty()) {
          throw new Exception(
              String.format(
                      "Wrong format, expecting the end of flie at the number %d, file '%s'.",
                      lineNum, fileName));                  
        }

      }
      
      postIndexComp();
      
      System.out.println("Finished loading context from file: " + fileName);
    } finally {    
      if (null != inp) inp.close();
    }
  }
  
  /**
   *  Pre-compute some values.
   */
  private void postIndexComp() {
    // Let's build a list of words & docs sorted by their IDs      
    buildWordListSortedById();
    buildDocListSortedById();
    // A mapping from word IDs to word entries.
    // MUST go after buildWordListSortedById()
    buildInt2WordEntry();
    
    mAvgDocLen = mTotalWordQty;
    mAvgDocLen /= mDocQty;
  }
    
  /**
   * @return an average document length.
   */
  public float getAvgDocLen() {
    return mAvgDocLen;
  }
  
  /**
   * @return a total number of documents.
   */
  public int getDocQty() {
    return mDocQty;
  }
  
  /**
   * 
   * @param word
   * @return a WordEntry of a word, or null if the word isn't found.
   */
  public WordEntry getWordEntry(String word) {
    return mStr2WordEntry.get(word);
  }
  
  /**
   * 
   * @return a WordEntry of a word represented by its ID. If the word
   *         with such ID doesn't exist the null is returned.
   */
  public WordEntry getWordEntry(int wordId) {
    WordEntryExt e = mInt2WordEntryExt.get(wordId);
    
    return e == null ? null : e.mWordEntry;
  }
  
  public String getWord(int wordId) {
    String res = null;
    
    WordEntryExt e = mInt2WordEntryExt.get(wordId);
    
    if (e != null) {
      return e.mWord;
    }
    
    return res;
  }

  public void save(String fileName) throws IOException {
    BufferedWriter out = null;
    
    try {
      out =    new BufferedWriter(
                  new OutputStreamWriter(
                      new FileOutputStream(fileName)));
      // 1. Write meta-info
      out.write(String.format("%d %d", mDocQty, mTotalWordQty));
      out.newLine();
      out.newLine();
      // 2. Write the dictionary
      for (WordEntryExt e: mWordEntSortById) {
        String    w    = e.mWord;
        WordEntry info = e.mWordEntry;
        out.write(String.format("%s\t%d:%d", w, info.mWordId, info.mWordFreq));
        out.newLine();
      }
      out.newLine();      
      // 3. Write the document entries
      for (DocEntryExt e : mDocEntSortById) {
        out.write(e.mId);
        out.newLine();
        DocEntry doc = e.mDocEntry;
        for (int i = 0; i < doc.mWordIds.length; ++i) {
          if (i > 0) out.write("\t");
          out.write(String.format("%d:%d", doc.mWordIds[i], doc.mQtys[i]));
        }
        out.newLine();
        if (doc.mWordIdSeq != null) {
          for (int i = 0; i < doc.mWordIdSeq.length; ++i) {
            if (i > 0) out.write(" ");
            out.write("" + doc.mWordIdSeq[i]);
          }
        } else {
          out.write(DOCLEN_QTY_PREFIX + doc.mDocLen);
        }
        out.newLine();
      }
      out.newLine();
    } finally {
      if (out != null) out.close();
    }
  }
  
  /**
   * Retrieves an existing document entry.
   * 
   * @param docId document id.
   * @return the document entry of the type {@link DocEntry} or null,
   *         if there is no document with the specified document ID.
   */
  public DocEntry getDocEntry(String docId) {
    return mStr2DocEntry.get(docId);
  }
  
  /**
   * Creates a document entry: a sequence of word IDs,
   * plus a list of words (represented again by their IDs)
   * with their frequencies of occurrence in the document.
   * This list is sorted by word IDs. Unknown words
   * have ID -1.
   * 
   * @param words             a list of document words.
   * @param bStoreWordIdSeq   if true, we memorize the sequence of word IDs, otherwise only a number of words (doc. len.)
   * 
   * @return a document entry.
   */
  public DocEntry createDocEntry(String[] words, boolean bStoreWordIdSeq) {
    // TreeMap guarantees that entries are sorted by the wordId
    TreeMap<Integer, Integer> wordQtys = new TreeMap<Integer, Integer>();        
    int [] wordIdSeq = new int[words.length];
    
    for (int i = 0; i < words.length; ++i) {
      String w = words[i];
      WordEntry wEntry = mStr2WordEntry.get(w);
      
      if (wEntry == null) {
        wEntry = UNKNOWN_WORD;
//        System.out.println(String.format("Warning: unknown token '%s'", w));
      }
        
      int wordId = wEntry.mWordId;
      
      wordIdSeq[i] = wordId;      
      Integer qty = wordQtys.get(wordId);
      if (qty == null) qty = 0;
      ++qty;
      wordQtys.put(wordId, qty);
    }
    
    DocEntry doc = new DocEntry(wordQtys.size(), wordIdSeq, bStoreWordIdSeq);
    
    int k =0;
    
    for (Map.Entry<Integer, Integer> e : wordQtys.entrySet()) {
      doc.mWordIds[k] = e.getKey();
      doc.mQtys[k]    = e.getValue();
      
      k++;
    }
    
    return doc;
  } 
  
  /**
   * Create a table where element with index i, keeps the 
   * probability of the word with ID=i; Thus we can efficiently
   * retrieve probabilities using word IDs.
   * 
   * @param voc     
   *            a GIZA vocabulary from which we take translation probabilities.
   * @return 
   *            a table where element with index i, keeps the 
   *            probability of the word with ID=i
   */
  public float[] createProbTable(GizaVocabularyReader voc) {
    if (mWordEntSortById.length == 0) return new float[0];
    int maxId = mWordEntSortById[mWordEntSortById.length-1].mWordEntry.mWordId;
    float res[] = new float[maxId + 1];
    
    for (WordEntryExt e : mWordEntSortById) {
      int id = e.mWordEntry.mWordId;
      res[id] = (float)voc.getWordProb(e.mWord);
    }
    
    return res;
  }
  
  /**
   * @return    a complete list of document entries: these are ordered
   *            in the order of their addition.
   */
  public ArrayList<DocEntryExt> getDocEntries() {
    return mDocEntInAdditionOrder;
  }
  
  void buildInt2WordEntry() {
    for (WordEntryExt e : mWordEntSortById) {
      mInt2WordEntryExt.put(e.mWordEntry.mWordId, e);
    }
  }
  
  public int getMaxWordId() { return mMaxWordId; }
   
  /**
   * @return an array containing all word IDs
   */
  public int [] getAllWordIds() {
    int [] res = new int [mWordEntSortById.length];
    for (int i = 0; i < mWordEntSortById.length; ++i)
      res[i] = mWordEntSortById[i].mWordEntry.mWordId;
    return res;
  }
   
  void buildWordListSortedById() {
    mWordEntSortById = new WordEntryExt[mStr2WordEntry.size()];
    
    int k = 0;
    for (Map.Entry<String, WordEntry> e : mStr2WordEntry.entrySet()) {
      mWordEntSortById[k++] = new WordEntryExt(e.getKey(), e.getValue());
      mMaxWordId = Math.max(mMaxWordId, e.getValue().mWordId);
    }
    Arrays.sort(mWordEntSortById);
  }
  
  void buildDocListSortedById() {
    mDocEntSortById = new DocEntryExt[mDocEntInAdditionOrder.size()];
    int k = 0;
    for (DocEntryExt e : mDocEntInAdditionOrder) {
      mDocEntSortById[k++] = e;
    }
    Arrays.sort(mDocEntSortById);
  }

  HashMap<String, WordEntry>    mStr2WordEntry = new HashMap<String, WordEntry>();
  HashMap<Integer,WordEntryExt> mInt2WordEntryExt = new HashMap<Integer, WordEntryExt>();
  HashMap<String, DocEntry>     mStr2DocEntry = new HashMap<String, DocEntry>();
  WordEntryExt[]                mWordEntSortById = null;
  DocEntryExt[]                 mDocEntSortById = null;
  ArrayList<DocEntryExt>        mDocEntInAdditionOrder = new ArrayList<DocEntryExt>();
  
  int   mDocQty = 0;
  int   mMaxWordId = 0;
  long  mTotalWordQty = 0;
  float mAvgDocLen = 0;
}
