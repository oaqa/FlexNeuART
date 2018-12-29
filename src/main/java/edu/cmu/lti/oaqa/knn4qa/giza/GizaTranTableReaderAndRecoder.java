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
package edu.cmu.lti.oaqa.knn4qa.giza;

import java.util.*;
import java.util.Map.Entry;
import java.io.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.openhft.koloboke.collect.map.hash.*;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VocabularyFilterAndRecoder;

/**
 * 
 * A helper class to read  translation table files produced by Giza or Giza++.
 * 
 * <p>Several important features:</p>
 * <li>It uses a filtering-and-recoding dictionary: the word IDs assigned to words
 *     by Giza are replaced with the IDs (using the recoding vocabulary). 
 * <li>Words not present in the recoding vocabulary are discarded. The respective
 *     translation rules are discarded as well.</li>
 * <li>Translation table entries are grouped by the source word ID.</li>
 * <li>During initialization, the class rescales translation probabilities so that 
 *      a probability of translating a word to itself is equal to a given value. 
 * </li>
 * <li>It is possible to swap source and target</li>
 * <li>We create a mapping from IDs to probabilities based on the source vocabulary.
 * </li>
 * <li>Probabilities related to zero-id word are currently ignored, 
 *      because they represent probabilities of 
 *      a spurious insertion. It's therefore expected that all word IDs in the
 *      filtering-and-recoding dictionary start with 1 (or a larger number).</li>
 * <li>We use Koloboke for better performance and smaller memory footprint.</li>
 * 
 * <p>To instantiate this class, one needs to read both the source and the target vocabulary 
 * files.</p>
 */
public class GizaTranTableReaderAndRecoder {
  private static final Logger logger = LoggerFactory.getLogger(GizaTranTableReaderAndRecoder.class);
  
  private static final int  REPORT_INTERVAL_QTY = 100000;
  
  private static final String BINARY_SUFFIX = ".bin";

  private static final int INIT_SIZE = 2 * 1024* 1024; // Most dictionaries will have < than this number of entries 

  public static String binaryFileName(String txtFileName) {
    return txtFileName + BINARY_SUFFIX;
  }
  
  /**
   * Constructor
   * 
   * @param flipTable
   *          reverse source/target translation tables as well as source/target word probabilities.
   * @param fileName
   *          input file name
   * @param filterAndRecoder
   *          used for filtering and recoding of string IDs.
   * @param vocSrc
   *          processed source vocabulary
   * @param vocDst
   *          processed target vocabulary
   * @param probSelfTran
   *          for rescaling purposes: a probability of translating a word into
   *          itself.
   * @param tranProbThreshold
   *          a threshold for the translation probability: records with values
   *          below the threshold are discarded. 
   * @throws Exception 
   */
  public GizaTranTableReaderAndRecoder(
                             boolean flipTable,
                             String fileName, 
                             VocabularyFilterAndRecoder filterAndRecoder,
                             GizaVocabularyReader vocSrc,
                             GizaVocabularyReader vocDst,
                             float probSelfTran,
                             float tranProbThreshold) throws Exception {
    mFilterAndRecoder = filterAndRecoder;
    
    mProbSelfTran = probSelfTran;
    if (probSelfTran >= 1 || probSelfTran < 0)
      throw 
      new Exception(
          String.format("Illegal self-translation probability %f", probSelfTran));
    
    BufferedReader fr = null;
    DataInputStream frBin = null;
    
    String binFileName = binaryFileName(fileName);
    
    if ((new File(binFileName).exists())) {
      System.out.println("Opening binary translation table.");
      frBin = new DataInputStream(new BufferedInputStream(CompressUtils.createInputStream(binFileName)));    
    } else {
      System.out.println("Opening text translation table.");
      fr = new BufferedReader(new InputStreamReader(
                                  CompressUtils.createInputStream(fileName)));
    }
    
    String line = null;
    
    int prevSrcId = -1;
    int recodedSrcId = -1;
    
    ArrayList<TranRecNoSrcId> tranRecs = new ArrayList<TranRecNoSrcId>();
    
    int wordQty = 0;
    
    int addedQty = 0;
    int totalQty = 0;
  
    for (totalQty = 0; ; ) {
      
      final GizaTranRec rec;
      
      // Skip empty lines
      
      if (fr != null) {
        line = fr.readLine();
        if (line == null)
          break;
        line = line.trim(); if (line.isEmpty()) continue;
        rec = new GizaTranRec(line);
      } else {
        try {
          int srcId = frBin.readInt();
          int dstId = frBin.readInt();
          float prob = frBin.readFloat();
          
          rec = new GizaTranRec(srcId, dstId, prob);
        } catch (EOFException e) {
          break;
        }
      }            

       ++totalQty;
       
      if (rec.mSrcId != prevSrcId) {
        if (rec.mSrcId < prevSrcId) {
          throw new Exception(
              String.format(
                  "Input file '%s' isn't sorted, encountered ID %d after %d ",
                   fileName, rec.mSrcId, prevSrcId));   
        }
        
        if (recodedSrcId > 0) { 
          procOneWord(recodedSrcId, tranRecs);
        }
        
        tranRecs.clear();
        ++wordQty;
      }
      
      if (totalQty % REPORT_INTERVAL_QTY == 0) {
        logger.info(String.format("Processed %d lines (%d source word entries) from '%s'", 
                                    totalQty, wordQty, fileName));
      }

      if (rec.mSrcId != prevSrcId) {
        recodedSrcId = -1;
        if (0 == rec.mSrcId) {
          recodedSrcId = 0;
        } else {        
          String wordSrc = vocSrc.getWord(rec.mSrcId);
          // wordSrc can be null, if vocSrc was also "filtered"
          if (wordSrc != null) {
            Integer tmpId = mFilterAndRecoder.getWordId(wordSrc);
            if (tmpId != null) {
              recodedSrcId = tmpId;
              float probSrc = (float)vocSrc.getWordProb(wordSrc);
              mSrcWordProb.put(recodedSrcId, probSrc);
            }
          }
        }
      }      
      prevSrcId = rec.mSrcId;

      if (recodedSrcId >=0 && 
          rec.mProb >= tranProbThreshold) {
        String wordDst = vocDst.getWord(rec.mDstId);
        // wordDst can be null, if vocDst was "filtered"
        if (wordDst != null) {
          Integer recodedDstId = mFilterAndRecoder.getWordId(wordDst);
          if (recodedDstId != null) {
            tranRecs.add(new TranRecNoSrcId(recodedDstId, rec.mProb));
            addedQty++;
            if (!mDstWordProb.containsKey((int)recodedDstId)) {
              float probDst = (float)vocDst.getWordProb(wordDst);
              mDstWordProb.put((int)recodedDstId, probDst);
            }
          }
        }
      }
    }
    
    if (recodedSrcId >= 0) { 
      procOneWord(recodedSrcId, tranRecs);
// Don't need to clear those any more
//      tranRecs.clear();      
      ++wordQty;      
    }
    logger.info(String.format("Processed %d source word entries from '%s'",
                               wordQty, fileName));
    logger.info(String.format("Loaded translation table from '%s' %d records out of %d", 
                               fileName, addedQty, totalQty));

    if (flipTable) {
      flipTranTable();
    }
  }
  
  private void procOneWord(int prevSrcId,
                           ArrayList<TranRecNoSrcId> tranRecs
                          ) throws Exception {
    /*
     * The input entries are always sorted by the first ID,
     * but not necessarily by the second ID, e.g., in the
     * flipped file (created by resorting by the second ID and flipping
     * columns 1 and 2).
     */
    Collections.sort(tranRecs);
    
    int prevId = -1;
    for (int i = 0; i < tranRecs.size(); ++i) {
      TranRecNoSrcId rec = tranRecs.get(i);
      if (rec.mDstId < prevId) {
        throw new Exception(String.format("Bug: not sorted by the second id, prevSrcId=%d, encountered %d after %d", prevSrcId, rec.mDstId, prevId));
      }
    }
    
    TranRecNoSrcId key = new TranRecNoSrcId(prevSrcId, 0);
    int indx = Collections.binarySearch(tranRecs, key);
    if (indx < 0) {
      tranRecs.add(-1-indx, key);
    }
    
    // Don't adjust in the case of spurious insertions (i.e., the source word ID is zero)
    float adjustMult = prevSrcId > 0 ? (1.0f - mProbSelfTran) : 1.0f;

    for (int i = 0; i < tranRecs.size(); ++i) {
      TranRecNoSrcId rec = tranRecs.get(i);
      rec.mProb *= adjustMult;
      if (rec.mDstId == prevSrcId) rec.mProb += mProbSelfTran;
    }
    
    mTranProb.put(prevSrcId, new GizaOneWordTranRecs(tranRecs));
  }
  
  /** 
   * Obtains a source-vocabulary word probability by the recoded word ID.
   * 
   * @param recodedWordId a recoded word identifier
   * @return              a probability of occurrences, or zero, if the word isn't found
   */
  public float getSourceWordProb(int recodedWordId) {
    return mSrcWordProb.get(recodedWordId);
  }
  
  /** 
   * Obtains a source-vocabulary word probability (using a string as input).
   * 
   * @param word          a string representation of the word.
   * @return              a probability of occurrences, or zero, if the word isn't found
   */
  public float getSourceWordProb(String word) {
    Integer recodedWordId = mFilterAndRecoder.getWordId(word);
    if (null == recodedWordId)  return 0;
    return mSrcWordProb.get((int)recodedWordId);
  }  

  /** 
   * Obtains a target-vocabulary word probability by the recoded word ID.
   * 
   * @param recodedWordId a recoded word identifier
   * @return              a probability of occurrences, or zero, if the word isn't found
   */
  public float getTargetWordProb(int recodedWordId) {
    return mDstWordProb.get(recodedWordId);
  }
  
  /** 
   * Obtains a target-vocabulary word probability (using a string as input).
   * 
   * @param word          a string representation of the word.
   * @return              a probability of occurrences, or zero, if the word isn't found
   */
  public float getTargetWordProb(String word) {
    Integer recodedWordId = mFilterAndRecoder.getWordId(word);
    if (null == recodedWordId)  return 0;
    return mDstWordProb.get((int)recodedWordId);
  }  
  
  
  
  /**
   * Obtains translation probabilities using a pair of word IDs rather than words themselves.
   * 
   * @param id1     id of the source string
   * @param id2     id of the target string
   * 
   * @return    a scaled translation probability, which is zero if there is no
   *            translation entry in the source table.
   */
   public float getTranProb(int id1, int id2) {
     float minProb = id1 == id2 ? mProbSelfTran : 0;
     GizaOneWordTranRecs recs = mTranProb.get(id1);
     
     if (null != recs) {
       int indx = Arrays.binarySearch(recs.mDstIds, id2);
       if (indx >= 0) {
         return Math.max(minProb, recs.mProbs[indx]);
       }
     }
     
     return minProb;
   }


   /**
    * Obtains a translation probability using a pair of strings.
    * 
    * @param s1      source string
    * @param s2      target string
    * @return    a scaled translation probability, which is zero if there is no
    *            translation entry in the source table.
    */
   public float getTranProb(String s1, String s2) {
     // Obtain source IDs
     float minProb = s1.equals(s2) ? mProbSelfTran : 0;
     Integer id1 = mFilterAndRecoder.getWordId(s1);
     Integer id2 = mFilterAndRecoder.getWordId(s2);
     if (null == id1 || null == id2) return minProb;
  
     return getTranProb(id1, id2);
   }
   
   /**
    * Obtains all translation probabilities for a given source word.
    * 
    * @return a reference to the object holding probabilities with respective IDs or null,
    *         if no record for a given wordId is found.
    */
   public GizaOneWordTranRecs getTranProbs(int wordId) {
     return mTranProb.get(wordId);
   }
   
   /**
    * The source becomes target and vice versa. 
    * 
    */
   private void flipTranTable() {
     logger.info("Flipping translation table started.");
     
     HashIntFloatMap   tmp = mSrcWordProb;     
     mSrcWordProb = mDstWordProb;
     mDstWordProb = tmp;
     
     HashIntObjMap<ArrayList<TranRecNoSrcId>>  newTranProb = 
         HashIntObjMaps.<ArrayList<TranRecNoSrcId>>newMutableMap(mTranProb.size());
     
     for (Entry<Integer, GizaOneWordTranRecs> e : mTranProb.entrySet()) {
       GizaOneWordTranRecs oldRec = e.getValue();
       int oldId = e.getKey();
       for (int k = 0; k < oldRec.mDstIds.length; ++k) {
         int newId = oldRec.mDstIds[k];
         ArrayList<TranRecNoSrcId> newRec = newTranProb.get(newId);
         
         if (null == newRec) {
           newRec = new ArrayList<TranRecNoSrcId>();
           newTranProb.put(newId, newRec);
         }
         newRec.add(new TranRecNoSrcId(oldId, oldRec.mProbs[k]));
       }
     }
     
     mTranProb = HashIntObjMaps.<GizaOneWordTranRecs>newMutableMap(mTranProb.size());
     
     for (Entry<Integer, ArrayList<TranRecNoSrcId>> e: newTranProb.entrySet()) {
       int id = e.getKey();
       
       ArrayList<TranRecNoSrcId>  tranRecs = e.getValue();
       Collections.sort(tranRecs);
       
       mTranProb.put(id,  new GizaOneWordTranRecs(tranRecs));
     }
     
     logger.info("Flipping translation table finished.");
   }
  
   private HashIntObjMap<GizaOneWordTranRecs>  mTranProb = 
                                            HashIntObjMaps.<GizaOneWordTranRecs>newMutableMap(INIT_SIZE);
   private HashIntFloatMap   mSrcWordProb = HashIntFloatMaps.newMutableMap(INIT_SIZE);
   private HashIntFloatMap   mDstWordProb = HashIntFloatMaps.newMutableMap(INIT_SIZE);
   private float mProbSelfTran = 0;
    
   private VocabularyFilterAndRecoder    mFilterAndRecoder;  
}

