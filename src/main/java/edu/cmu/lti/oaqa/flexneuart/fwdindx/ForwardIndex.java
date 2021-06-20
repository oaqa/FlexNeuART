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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import edu.cmu.lti.oaqa.flexneuart.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Joiner;

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
 * A forward index. 
 * 
 * <p>A base abstract class for the forward index that is created for <b>one field</b>.
 * The forward index is created from a file create by a parsing pipeline.
 * The file can contain either text JSON entries or binary BSON entries.
 * There are three major index types: parsed text (two-subtypes), raw/unmodified text,
 * binary entries.</p>
 * 
 * <p>The raw text or binary index merely keeps all entries in the unparsed form.
 * Text entries are being compressed, but binary entries are not.
 * For the parsed text, we assume that tokens are space-separated. 
 * We then compile all field-specific unique (space-separated) into a dictionary.
 * </p>>
 * 
 * <p>
 * In addition, we compute some statistics for each field in the parsed index:
 * </p>
 * <ol> 
 *  <li>An average number of words per document;
 *  <li>The total number of documents;
 *  <li>An number of documents where each word occur;
 * </ol>  
 * 
 * <p><b>NOTE:</b> word IDs start from 1.</p>
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class ForwardIndex {

  private static final Logger logger = LoggerFactory.getLogger(ForwardIndex.class);
  
  public enum ForwardIndexBackendType {
    mapdb,
    inmem,
    flatdata,
    unknown // to use as an indicator that a string entry doesn't correspond to the forward index time
  }
  
  public enum ForwardIndexFieldType {
    binary,
    textRaw,
    parsedBOW,
    parsedText,
    unknown // to use as an indicator that a string entry doesn't correspond to the forward index time
  }
  
  public static final WordEntry UNKNOWN_WORD = new WordEntry(Const.UNKNOWN_WORD_ID);
  public static final int MIN_WORD_ID = 1;
  protected static final int PRINT_QTY = 10000;
  
  protected HashMap<String, WordEntry> mStr2WordEntry = new HashMap<String, WordEntry>();
  
  /**
   * Is the index type raw text? This functions returns a meaningful value only if the 
   * instance was created for writing, or after we read the header from the disk.
   * 
   * @return true when the index type is not raw, or we just created
   *              an instance for reading, but didn't actually read contents.
   */
  public boolean isTextRaw() { return mIndexFieldType == ForwardIndexFieldType.textRaw; }
  
  /**
   * Is it a binary field? This functions returns a meaningful value only if the 
   * instance was created for writing, or after we read the header from the disk.
   * 
   * @return true when the index type is not raw, or we just created
   *              an instance for reading, but didn't actually read contents.
   */
  public boolean isBinary() { return mIndexFieldType == ForwardIndexFieldType.binary; }
  
  /**
   * Is it a parsed field? This functions returns a meaningful value only if the 
   * instance was created for writing, or after we read the header from the disk.
   * 
   * @return true when the index type is not raw, or we just created
   *              an instance for reading, but didn't actually read contents.
   */
  public boolean isParsed() { 
    return mIndexFieldType == ForwardIndexFieldType.parsedBOW || mIndexFieldType == ForwardIndexFieldType.parsedText; 
  }
  
  /**
   * Retrieves an existing parsed document entry. 
   * 
   * @param docId document id.
   * @return the document entry of the type {@link DocEntryParsed} or null,
   *         if there is no document with the specified document ID.
   *         
   * @throws An exception if there is a retrieval error, or if this is not a parsed field.
   */
  public abstract DocEntryParsed getDocEntryParsed(String docId) throws Exception;
  
  /**
   * Retrieves an existing text document entry. Raw means that the entry is stored
   * as-is without any parsing.
   * 
   * @param docId document id.
   * @return raw document string or null,
   *         if there is no document with the specified document ID.
   *         
   * @throws An exception if there is a retrieval error, or if the field index type is not "raw text".
   */
  public abstract String getDocEntryTextRaw(String docId) throws Exception;
  
  /**
   * Retrieves an existing binary document entry.
   * 
   * @param docId document id.
   * @return binary document representation or null,
   *         if there is no document with the specified document ID.
   *         
   * @throws An exception if there is a retrieval error, or if the field index type is not "binary".
   */
  public abstract byte[] getDocEntryBinary(String docId) throws Exception;
  
  /**
   * Convert an index <b>backend</b> type to the corresponding enum.
   * 
   * @param type a string type
   * @return the corresponding enum value or the unknown value if the string isn't recognized
   */
  public static ForwardIndexBackendType getIndexBackendType(String type) {
    for (ForwardIndexBackendType itype : ForwardIndexBackendType.values()) {
      if (itype.toString().compareToIgnoreCase(type) == 0) {
        return itype;
      }
    }
    return ForwardIndexBackendType.unknown;
  }
  
  /**
   * Convert an index <b>field</b> type to the corresponding enum.
   * 
   * @param type a string type
   * @return the corresponding enum value or the unknown value if the string isn't recognized
   */
  public static ForwardIndexFieldType getIndexFieldType(String type) {
    for (ForwardIndexFieldType itype : ForwardIndexFieldType.values()) {
      if (itype.toString().compareToIgnoreCase(type) == 0) {
        return itype;
      }
    }
    return ForwardIndexFieldType.unknown;
  }
  
  public String getIndexFieldType() {
    return mIndexFieldType.toString();
  }
  
  public static String getIndexBackendTypeList() {
    Joiner   joinOnComma  = Joiner.on(',');
    ArrayList<String> parts = new ArrayList<String>();
    for (ForwardIndexBackendType itype : ForwardIndexBackendType.values()) {
      if (itype != ForwardIndexBackendType.unknown) {
        parts.add(itype.toString());
      }
    }
    
    return joinOnComma.join(parts);
  }

  public static String getIndexFieldTypeList() {
    Joiner   joinOnComma  = Joiner.on(',');
    ArrayList<String> parts = new ArrayList<String>();
    for (ForwardIndexFieldType itype : ForwardIndexFieldType.values()) {
      if (itype != ForwardIndexFieldType.unknown) {
        parts.add(itype.toString());
      }
    }
    return joinOnComma.join(parts);
  }
  
  /**
   * Create an index file instance that can be used to create/save index.
   * 
   * @param filePrefix  a prefix of the index file/directories
   * @param indexBackendType a type of the index backend (text, lucene, mapdb)
   * @param indexFieldType a type of the index field (parsed, binary, raw text)
   * 
   * @return a  ForwardIndex sub-class instance
   * @throws IOException
   */
  public static ForwardIndex createWriteInstance(String filePrefix,
  												ForwardIndexBackendType indexBackendType,
  												ForwardIndexFieldType indexFieldType) throws IOException {
    return createInstance(filePrefix, indexBackendType, indexFieldType);
  }

  /**
   * Create a readable instance of the index. The function does not
   * need to know the type of the field (parsed, raw text, binary) at
   * this point. The type of the field will be obtained from the header.
   * 
   * @param filePrefix a prefix of the index file/directories
   * @return a ForwardIndex sub-class instance
   * @throws Exception
   */
  public static ForwardIndex createReadInstance(String filePrefix) throws Exception {    
    // If for some weird reason more than one index was created, we will try to use the first one
    ForwardIndex res = null;
    
    for (ForwardIndexBackendType itype : ForwardIndexBackendType.values()) {
      String indexPrefixFull = getIndexPrefix(filePrefix, itype);  
      File indexDirOrFile = new File(indexPrefixFull);
      
      if (indexDirOrFile.exists()) {
        res = createInstance(filePrefix, itype, ForwardIndexFieldType.unknown);
        break;
      }
    }
    
    if (null == res) {
      throw new Exception("No index found at location: " + filePrefix);
    }
    
    res.readIndex();
    return res;
  }
  
  abstract public String[] getAllDocIds();
   
  /**
   * Retrieves a previously stored index.
   * 
   * @param fileName the file generated by the function {@link #save(String)}.
   */
  abstract public void readIndex() throws Exception;
  
  public abstract void saveIndex() throws IOException;
  
  protected abstract void sortDocEntries();
  
  /**
   *  Pre-compute some values.
   */
  protected void postIndexComp() {
    // Let's build a list of words & docs sorted by their IDs      
    buildWordListSortedById();
    
    // A mapping from word IDs to word entries.
    // MUST go after buildWordListSortedById()
    buildInt2WordEntry();
    
    sortDocEntries();
    
    if (mDocQty > 0) {
      mAvgDocLen = mTotalWordQty;
      mAvgDocLen /= mDocQty;
    }
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

  protected void writeHeader(BufferedWriter out) throws IOException {
    // 1. Write the index type
    out.write(mIndexFieldType.toString());
    // 2. Write meta-info, if the number of words is negative
    //    this is an indicator that the index is raw, not parsed
    out.newLine();
    out.write(String.format("%d %d", mDocQty, isParsed() ? mTotalWordQty : -1));
    out.newLine();
    out.newLine();
    // 3. Write the dictionary (if the index is not raw)
    if (isParsed()) {
      for (WordEntryExt e: mWordEntSortById) {
        String    w    = e.mWord;
        WordEntry info = e.mWordEntry;
        out.write(String.format("%s\t%d:%d", w, info.mWordId, info.mWordFreq));
        out.newLine();
      }
    }
    out.newLine();
  }



  /***
   * This function constructs a textual representation of parsed document/query entry.
   * This function needs a positional index. If the input is null, we return an empty string.
   * 
   * @param e a parsed entry
   * @return parsed entry text
   * @throws Exception
   */
  public String getDocEntryParsedText(DocEntryParsed e) throws Exception {
    if (e == null) {
      return "";
    }
    StringBuffer sb = new StringBuffer();

    if (e.mWordIdSeq == null) {
        throw new Exception("Positional information is missing in the index!");
    }
    
    for (int i = 0; i < e.mWordIdSeq.length; ++i) {
      if (i > 0) {
        sb.append(' ');
      }
      int wid = e.mWordIdSeq[i];
      String w = getWord(wid);
      if (w == null) {
        throw new Exception("Looks like bug or inconsistent data, no word for word id: " + wid);
      }
      sb.append(w);
    }
    
    return sb.toString();
  }
  
  /**
   * Retrieves an existing parsed document entry and constructs a textual representation.
   * This function needs a positional index.
   * 
   * @param docId document id.
   * @return the document text or null,
   *         if there is no document with the specified document ID.
   *         
   * @throws An exception if there is a retrieval error, or if the index type is raw.
   */
  public String getDocEntryParsedText(String docId) throws Exception {
    DocEntryParsed e = getDocEntryParsed(docId);
    if (e == null) {
      return null;
    }
    return getDocEntryParsedText(e);
  }

  /**
   * Creates a parsed document entry: a sequence of word IDs,
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
  public DocEntryParsed createDocEntryParsed(String[] words, boolean bStoreWordIdSeq) {
      // TreeMap guarantees that entries are sorted by the wordId
      TreeMap<Integer, Integer> wordQtys = new TreeMap<Integer, Integer>();        
      int [] wordIdSeq = new int[words.length];
      
      for (int i = 0; i < words.length; ++i) {
        String w = words[i];
        WordEntry wEntry = mStr2WordEntry.get(w);
        
        if (wEntry == null) {
          wEntry = UNKNOWN_WORD;
        }
          
        int wordId = wEntry.mWordId;
        
        wordIdSeq[i] = wordId;      
        Integer qty = wordQtys.get(wordId);
        if (qty == null) qty = 0;
        ++qty;
        wordQtys.put(wordId, qty);
      }
      
      DocEntryParsed doc = new DocEntryParsed(wordQtys.size(), wordIdSeq, bStoreWordIdSeq);
      
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

  void buildInt2WordEntry() {
    for (WordEntryExt e : mWordEntSortById) {
      mInt2WordEntryExt.put(e.mWordEntry.mWordId, e);
    }
  }

  public int getMaxWordId() { return mMaxWordId; }
  
  // addDocEntry* functions are not supposed to be thread-safe
  // the indexing app shouldn't be multi-threaded
  protected abstract void addDocEntryParsed(String docId, DocEntryParsed doc) throws IOException;
  protected abstract void addDocEntryTextRaw(String docId, String docText) throws IOException;
  protected abstract void addDocEntryBinary(String docId, byte [] docBin) throws IOException;

  protected abstract void initIndex() throws IOException;
  
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

  HashMap<Integer,WordEntryExt> mInt2WordEntryExt = new HashMap<Integer, WordEntryExt>();
  WordEntryExt[] mWordEntSortById = null;

  protected ForwardIndexFieldType mIndexFieldType = ForwardIndexFieldType.unknown;
  protected boolean mStoreWordIdSeq = false;
  protected int     mDocQty = 0;
  int               mMaxWordId = 0;
  protected long    mTotalWordQty = 0;
  float             mAvgDocLen = 0;

  /**
   * Read the text-only header (which includes vocabulary info) of the forward file. If the format changes,
   * please, update/sync with (and possibly make them reuse the same shared common function, so we have no copy-paste)
   * {@link FrequentIndexWordFilterAndRecoder}.
   * 
   * @param fileName  input file name (for info purposes only)
   * @param inp  the actual input file opened
   * 
   * @return a line number read plus one
   * 
   * @throws IOException
   * @throws Exception
   */
  protected int readHeader(String fileName, BufferedReader inp) throws IOException, Exception {
    String indexFieldTypeStr = inp.readLine();
    
    mIndexFieldType = getIndexFieldType(indexFieldTypeStr);
    
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
    
    // Read the dictionary if the index is parsed
    int lineNum = 3;
    line = inp.readLine();
    
    if (mTotalWordQty < 0) {
      if (isParsed()) {
        throw new Exception("Inconsistent data: index type is: " + getIndexFieldType() + " but word information is missing.");
      }
    }
    
    for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
      if (!isParsed()) {
        throw new Exception(
            String.format("Inconsistent data: word information is present in the index type %s, line %d, file %s",
                getIndexFieldType(), lineNum, fileName));
      }
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
    return lineNum;
  }
  
  private static String getIndexPrefix(String filePrefix, ForwardIndexBackendType indexBackendType) {
    if (indexBackendType == ForwardIndexBackendType.unknown) {
      throw new RuntimeException("getIndexPrefix shouldn't be called with backend index type: unknown!");
    }
    return filePrefix + "." + indexBackendType.toString();
  }
  
  // ForwardIndexFieldType.unknown index type is used to create a read instance
  private static ForwardIndex createInstance(String filePrefix,
  																					 ForwardIndexBackendType indexBackendType,
  																					 ForwardIndexFieldType indexFieldType) throws IOException {
    String indexPrefixFull = getIndexPrefix(filePrefix, indexBackendType);
    ForwardIndex res = null;
    
    switch (indexBackendType) {
      case mapdb:    res = new ForwardIndexBinaryMapDb(filePrefix, indexPrefixFull); break;
      case inmem:    res = new ForwardIndexTextInMem(indexPrefixFull); break;
      case flatdata: res = new ForwardIndexBinaryFlatFileData(filePrefix, indexPrefixFull); break;
      case unknown: throw new RuntimeException("getIndexPrefix shouldn't be called with index type: unknown!");
    }
    
    res.mIndexFieldType = indexFieldType;
    res.mStoreWordIdSeq = indexFieldType == ForwardIndexFieldType.parsedText;
                         
    return res;    
  }

  
  /**
   * Creates an index from one or more files (for a given field name).
   * 
   * @param fieldName         the name of the field (as specified in the SOLR index-file)
   * @param fileNames         an array of files from which the index is created
   * @param maxNumRec         the maximum number of records to process
   * 
   * @throws Exception 
   */
  public void createIndex(String fieldName, String[] fileNames, 
                         int maxNumRec) throws Exception {    
    mDocQty       = 0;
    mTotalWordQty = 0;
    
    initIndex();
    
    long totalUniqWordQty = 0; // sum the number of uniq words per document (over all documents)
    
    logger.info("Creating a new forward index, maximum # of docs to process: " + maxNumRec);
    
    for (String fileName : fileNames) {    
      try (DataEntryReader inp = new DataEntryReader(fileName)) {
 
        DataEntryFields  dataEntry = null;
        
        for (; mDocQty < maxNumRec && ((dataEntry = inp.readNext()) != null) ;) {
          ++mDocQty;
          
          if (dataEntry.mEntryId == null) {
            logger.warn(String.format("No entry/doc ID entry #%d ignoring", mDocQty));
            continue;
          }
          
          String text = null;
          
          if (isParsed() || isTextRaw()) {
              text = dataEntry.getString(fieldName);

              if (text == null) text = "";
              if (text.isEmpty()) {
                logger.warn(String.format("Warning: empty field '%s' for document '%s'",
                                          fieldName, dataEntry.mEntryId));
              }
          }

          if (isTextRaw()) {
            addDocEntryTextRaw(dataEntry.mEntryId, text);
          } else if (isBinary()) {
            byte [] data = dataEntry.getBinary(fieldName);
            if (data == null) {
              logger.warn(String.format("Warning: empty field '%s' for document '%s'",
                                         fieldName, dataEntry.mEntryId));
            } else {
              addDocEntryBinary(dataEntry.mEntryId, data);
            }
          } else {         
            // If the string is empty, the array will contain an empty string, but we don't want this
            text=text.trim();
            String words[];
   
            DocEntryParsed doc;
            
            words = text.isEmpty() ? new String[0] : text.split("\\s+");
            // First obtain word IDs for unknown words
            for (int i = 0; i < words.length; ++i) {
              String w = words[i];
              WordEntry wEntry = mStr2WordEntry.get(w);
              if (null == wEntry) {
                wEntry = new WordEntry(MIN_WORD_ID + mStr2WordEntry.size());
                mStr2WordEntry.put(w, wEntry);
              }
            }
            doc = createDocEntryParsed(words, mStoreWordIdSeq);
            addDocEntryParsed(dataEntry.mEntryId, doc);
            HashSet<String> uniqueWords = new HashSet<String>();
            for (String w : words)
              uniqueWords.add(w);
            // Let's update word co-occurrence statistics
            for (String w : uniqueWords) {
              WordEntry wEntry = mStr2WordEntry.get(w);
              wEntry.mWordFreq++;
            } 
            mTotalWordQty += words.length;
            totalUniqWordQty += doc.mQtys.length;
          }
          
          if (mDocQty % PRINT_QTY == 0) {
            logger.info("Processed " + mDocQty + " documents");
            System.gc();
          }
        }
        
        postIndexComp();
        
        logger.info("Finished processing file: " + fileName);
        
        if (mDocQty >= maxNumRec) break;
      }
    }
    
    logger.info("Final statistics: ");
    logger.info(
        String.format("Number of documents %d, total number of words %d, average reduction due to keeping only unique words %f",
                      mDocQty, mTotalWordQty, 
                      // When total # of words is zero, without max we would be dividing by zero
                      ((double)mTotalWordQty)/Math.max(1,totalUniqWordQty)));
  }
  
   
  
}