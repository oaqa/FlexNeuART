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

package edu.cmu.lti.oaqa.knn4qa.embed;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Formatter;
import java.util.Map.Entry;

import org.apache.tools.ant.taskdefs.ManifestTask.Mode;

import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.Vector.Norm;
import no.uib.cipr.matrix.sparse.SparseVector;
import net.openhft.koloboke.collect.map.hash.HashIntObjMap;
import net.openhft.koloboke.collect.map.hash.HashIntObjMaps;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.FrequentIndexWordFilterAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.WordEntry;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaOneWordTranRecs;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.simil_func.TFIDFSimilarity;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.VocabularyFilterAndRecoder;


class WordIdProb implements Comparable<WordIdProb> {
  public final int    mWordId;
  public final double mProb;
  public WordIdProb(int wordId, double prob) {
    this.mWordId = wordId;
    this.mProb = prob;
  }
  @Override
  public int compareTo(WordIdProb o) {
    return mWordId - o.mWordId; 
  }  
};

class WordIdVals implements Comparable<WordIdVals> {
  public final int    mWordId;
  public final double mVal;
  public final double mValTFxIDF;
  public final double mValTFProb;
  public WordIdVals(int wordId, double val, double valTFxIDF, double valTFProb) {
    mWordId = wordId;
    mVal = val;
    mValTFxIDF = valTFxIDF;
    mValTFProb = valTFProb;
  }
  @Override
  public int compareTo(WordIdVals o) {
    return mWordId - o.mWordId; 
  }  
};

public class SparseEmbeddingReaderAndRecorder {
  
  /**
   * Saving sparse word embeddings.
   * 
   * @param fieldIndex
   *          in-memory forward index
   * @param fileName
   *          output file name
   * @param dict
   *          the model itself in the form of a mapping wordId => embedding
   * @param maxDigit
   *          the maximum # of digits to print
   * @throws IOException
   */
  public static void saveDict(ForwardIndex           fieldIndex,
                              String                      fileName, 
                              HashIntObjMap<SparseVector> dict,
                              int                         maxDigit) throws IOException {
    BufferedWriter out = new BufferedWriter(
                              new OutputStreamWriter(CompressUtils.createOutputStream(fileName)));
    
    String outFormat= " %d:%." + maxDigit + "e";
    
    for (Entry<Integer, SparseVector> e: dict.entrySet()) {
      out.write(fieldIndex.getWord(e.getKey()));
      for (VectorEntry ve : e.getValue()) {
        Formatter f = new Formatter();
        f.format(outFormat, ve.index(), ve.get());
        out.write(f.toString());
        f.close();
      }
      out.newLine();
    }
    
    out.close();    
  }
  
  public static HashIntObjMap<SparseVector> readDict(ForwardIndex    fieldIndex,
                                                     String               fileName) throws Exception {
    BufferedReader in = new BufferedReader(
                                new InputStreamReader(CompressUtils.createInputStream(fileName)));
    
    HashIntObjMap<SparseVector> res = HashIntObjMaps.<SparseVector>newMutableMap(fieldIndex.getMaxWordId() + 1);
    
    String line;
    int lineNum = 0;
    while ((line = in.readLine()) != null) {
      ++lineNum;
      line = line.trim();
      if (line.isEmpty()) continue;
      String parts[] = line.split("\\s+");
      if (parts.length < 1) {
        throw new Exception("Bug: not enough elements in line: " + lineNum + " file: '" + fileName + "'");
      }
      String word = parts[0];
      WordEntry e = fieldIndex.getWordEntry(word);
      if (e == null) {
        throw new Exception("The word embedding was computed for a different collection, we encountered an unknown word '" + word + 
                            "' :" + lineNum + " file: '" + fileName + "'");
      }
      int     index[] = new int[parts.length-1];
      double  data [] = new double[parts.length-1];
      for (int i = 1; i < parts.length; ++i) {
        String tmp[] = parts[i].split(":");
        if (tmp.length != 2) {
          throw new Exception("Wrong format of the part '" + parts[i] + "' in line: " + lineNum + " file: '" + fileName + "'");
        }
        index[i-1]=Integer.parseInt(tmp[0]);
        data [i-1]=Float.parseFloat(tmp[1]);
      }
      res.put(e.mWordId, new SparseVector(fieldIndex.getMaxWordId() + 1, index, data, false));
    }
    
    return res;
  }
  
  public static HashIntObjMap<SparseVector> createTranVecDict(ForwardIndex                 fieldIndex,
                                                              FrequentIndexWordFilterAndRecoder filter, 
                                                              float                             minProb, 
                                                              GizaTranTableReaderAndRecoder     answToQuestTran) throws Exception {
    HashIntObjMap<SparseVector> res = HashIntObjMaps.<SparseVector>newMutableMap(fieldIndex.getMaxWordId() + 1);
    
    for (int srcWordId : fieldIndex.getAllWordIds()) 
    if (filter.checkWordId(srcWordId)) {
      res.put(srcWordId, createOneTranVector(fieldIndex, minProb, answToQuestTran, srcWordId));
    }
    
    return res;
  }
  
  // TODO his would certainly work for a symmetric computation of translation tables
  // but otherwise I am not sure if source/target are used here correctly. 
  public static HashIntObjMap<SparseVector> nextOrderDict(HashIntObjMap<SparseVector>   dict, 
                                                          ForwardIndex             fieldIndex,
                                                          float                         minProb, 
                                                          GizaTranTableReaderAndRecoder answToQuestTran) throws Exception {
    HashIntObjMap<SparseVector> res = HashIntObjMaps.<SparseVector>newMutableMap(fieldIndex.getMaxWordId() + 1);
    
    for (int srcWordId : dict.keySet()) {
      SparseVector    newVal = new SparseVector(fieldIndex.getMaxWordId() + 1); 
      GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(srcWordId);                 
      if (null != tranRecs) {                
        for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
          int     dstWordId = tranRecs.mDstIds[k];
          double  prob      = tranRecs.mProbs[k];
          SparseVector v = dict.get(dstWordId).copy();
          if (v != null) {
            newVal.add(prob, v);
          }
        }
      }
      double norm = newVal.norm(Norm.One);
      if (norm > 0) newVal.scale(1.0/norm);
      // Get rid of small values
      double[] data = newVal.getData();
      for (int i = 0; i < newVal.getUsed(); ++i) {
        if (data[i] < minProb) {
          data[i]=0.0;
        }
      }
      newVal.compact();
      res.put(srcWordId, newVal);
    }
    
    return res;
  }
  
  /**
   * Create an un-normalzied sparse word embedding based on translation probabilities.
   * 
   * @param fieldIndex        
   *            an in-memory forward index.
   * @param minProb           
   *            ignore if the translation probability is smalelr than this one.
   * @param answToQuestTran   
   *            a provider of translation tables (computed originally by GIZA).
   * @param srcWordId         
   *            a source word ID
   * @return    an un-normalzied sparse word embedding based on translation probabilities.
   *            
   * @throws Exception
   */
  public static SparseVector createOneTranVector(ForwardIndex             fieldIndex,
                                                 float                         minProb, 
                                                 GizaTranTableReaderAndRecoder answToQuestTran,
                                                 int srcWordId) throws Exception {
    ArrayList<WordIdProb>   data = new ArrayList<WordIdProb>();
    if (srcWordId >= 0) {     
      GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(srcWordId);
      
      if (null != tranRecs) {
        for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
          int dstWordId = tranRecs.mDstIds[k];
          double val = tranRecs.mProbs[k];
          if (val >= minProb) {
            data.add(new WordIdProb(dstWordId, val));
          }
        }
      }
    }
      
    WordIdProb dataArr[] = new WordIdProb[data.size()];
    dataArr = data.toArray(dataArr);
    Arrays.sort(dataArr);
    
    int     wordIds[] = new int[data.size()];
    double  vals[] = new double[data.size()];

    int indx = -1;
    int prevWordId = -1;
    for (WordIdProb e: dataArr) {
      if (e.mWordId != prevWordId) indx++;
      else 
        throw new Exception("Bug, repeating wordId " + prevWordId + ", translation record for wordId=" + srcWordId);
      wordIds[indx] = e.mWordId;
      vals[indx] += e.mProb;
      prevWordId = e.mWordId;
    }
    
    return new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, vals, false);
  }
  
  /**
   * Create a composite word-embedding based on translation probabilities.
   * The idea is taken from the following paper (though the implementation is a bit different):
   * Higher-order Lexical Semantic Models for Non-factoid Answer Reranking.
   * Fried, et al. 2015.
   *
   * @param fieldIndex        an in-memory field index
   * @param minProb           a minimum translation probability
   * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
   * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++)
   * @param similObj          an object that computes similarity (need this to compute IDF).
   * @param entry             a document/query entry.
   * @param bMultByTF         when combining translation probabilities, additionally multiply by term frequency (TF)
   * @param bMultByProb       when combining translation probabilities, additionally multiply by term probability.
   * @param bMultByIDF        when combining translation probabilities, additionally multiply by IDF. 
   * 
   * @return
   */
  public static SparseVector createTranBasedCompositeWordEmbed(ForwardIndex             fieldIndex,
                                               float minProb, 
                                               float[] fieldProbTable,
                                               GizaTranTableReaderAndRecoder answToQuestTran,                                               
                                               TFIDFSimilarity            similObj,
                                               DocEntry                      entry,
                                               boolean                       bMultByTF,
                                               boolean                       bMultByProb,
                                               boolean                       bMultByIDF) {
    ArrayList<WordIdProb>   data = new ArrayList<WordIdProb>();
    
    
    for (int i = 0; i < entry.mWordIds.length; ++i) {
      int srcWordId = entry.mWordIds[i];
      
      if (srcWordId < 0) continue; // ignore OOV words
      
      GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(srcWordId);
      
      if (null != tranRecs) {
        //System.out.print(fieldIndex.getWord(srcWordId));
        for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
          int dstWordId = tranRecs.mDstIds[k];
          double val = tranRecs.mProbs[k];
          if (val >=  minProb) {
            if (bMultByTF)    val *= entry.mQtys[i];
            if (bMultByIDF)   val *= similObj.getIDF(fieldIndex, srcWordId);
            if (bMultByProb)  val *= fieldProbTable[srcWordId];
            data.add(new WordIdProb(dstWordId, val));
            //System.out.print(" " + dstWordId + ":" + val);
          }
        }
        //System.out.println();
      }        
    }
    WordIdProb dataArr[] = new WordIdProb[data.size()];
    dataArr = data.toArray(dataArr);
    Arrays.sort(dataArr);
    int uniqQty = 0;
    int prevWordId = -1;
    for (WordIdProb e: dataArr) {
      if (e.mWordId != prevWordId) uniqQty++;
      prevWordId = e.mWordId;
    }
    int     wordIds[] = new int[uniqQty];
    double  vals[] = new double[uniqQty];

    int indx = -1;
    prevWordId = -1;
    for (WordIdProb e: dataArr) {
      if (e.mWordId != prevWordId) indx++;
      wordIds[indx] = e.mWordId;
      vals[indx] += e.mProb;
      prevWordId = e.mWordId;
    }
    
    return new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, vals, false);
  }
  

  /**
   * Create three composite word-embedding based on translation probabilities.
   * The idea is taken from the following paper (though the implementation is a bit different):
   * Higher-order Lexical Semantic Models for Non-factoid Answer Reranking.
   * Fried, et al. 2015.
   *
   * @param fieldIndex        an in-memory field index
   * @param minProb           a minimum translation probability
   * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
   * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++)
   * @param similObj          an object that computes similarity (need this to compute IDF).
   * @param entry             a document/query entry.
   * 
   * @return
   */
  public static TranBasedWordEbmeddings createTranBasedCompositeWordEmbedings(ForwardIndex             fieldIndex,
                                               float minProb, 
                                               float[] fieldProbTable,
                                               GizaTranTableReaderAndRecoder answToQuestTran,                                               
                                               TFIDFSimilarity            similObj,
                                               DocEntry                      entry) {
    ArrayList<WordIdVals>   data = new ArrayList<WordIdVals>();
    
    for (int i = 0; i < entry.mWordIds.length; ++i) {
      int srcWordId = entry.mWordIds[i];
      
      if (srcWordId < 0) continue; // ignore OOV words
      
      GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(srcWordId);
      
      if (null != tranRecs) {
        for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
          int dstWordId = tranRecs.mDstIds[k];
          double prob = tranRecs.mProbs[k];
          if (prob >=  minProb) {
            data.add(new WordIdVals(dstWordId, prob, 
                                               prob * entry.mQtys[i] * similObj.getIDF(fieldIndex, srcWordId),
                                               prob * entry.mQtys[i] * fieldProbTable[srcWordId]));
          }
        }
      }        
    }
    
    WordIdVals dataArr[] = new WordIdVals[data.size()];
    dataArr = data.toArray(dataArr);
    Arrays.sort(dataArr);
    int uniqQty = 0;
    int prevWordId = -1;
    for (WordIdVals e: dataArr) {
      if (e.mWordId != prevWordId) uniqQty++;
      prevWordId = e.mWordId;
    }
    int     wordIds[]     = new int[uniqQty];
    double  vals[]        = new double[uniqQty];
    double  valsTFxIDF[]  = new double[uniqQty];
    double  valsTFProb[]  = new double[uniqQty];
    
    int indx = -1;
    prevWordId = -1;
    for (WordIdVals e: dataArr) {
      if (e.mWordId != prevWordId) indx++;
      wordIds[indx] = e.mWordId;
      vals[indx]       += e.mVal;
      valsTFxIDF[indx] += e.mValTFxIDF;
      valsTFProb[indx] += e.mValTFProb;
      prevWordId = e.mWordId;
    }
    
    return new TranBasedWordEbmeddings( 
        new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, vals, false),
        new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, valsTFxIDF, false),
        new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, valsTFProb, false));
  }

  /**
   * This function "composes" an L1-normalized word embedding by summing up individual
   * sparse word vectors.
   * 
   * @param fieldIndex      an in-memory forward index.
   * @param model           a mapping from words to embeddings
   * @param entry           a document/query entry.
   * @return
   */
  public static SparseVector createCompositeWordEmbed(ForwardIndex             fieldIndex,
                                                      HashIntObjMap<SparseVector>   model, 
                                                      DocEntry                      entry) {

    ArrayList<WordIdProb>   data = new ArrayList<WordIdProb>();
        
    for (int i = 0; i < entry.mWordIds.length; ++i) {
      int srcWordId = entry.mWordIds[i];
      
      if (srcWordId < 0) continue; // ignore OOV words
      
      SparseVector vec = model.get(srcWordId);
      
      if (null != vec) {
        for (VectorEntry e: vec) {
          data.add(new WordIdProb(e.index(), e.get()));
        }
      }        
    }
    
    WordIdProb dataArr[] = new WordIdProb[data.size()];
    dataArr = data.toArray(dataArr);
    Arrays.sort(dataArr);
    int uniqQty = 0;
    int prevWordId = -1;
    for (WordIdProb e: dataArr) {
      if (e.mWordId != prevWordId) uniqQty++;
      prevWordId = e.mWordId;
    }
    int     wordIds[] = new int[uniqQty];
    double  vals[] = new double[uniqQty];

    int indx = -1;
    prevWordId = -1;
    for (WordIdProb e: dataArr) {
      if (e.mWordId != prevWordId) indx++;
      wordIds[indx] = e.mWordId;
      vals[indx] += e.mProb;
      prevWordId = e.mWordId;
    }
    
    SparseVector res = new SparseVector(fieldIndex.getMaxWordId()+1, wordIds, vals, false);
    double norm = res.norm(Norm.One);
    if (norm > 0) res.scale(1.0/norm);
    return res;
  }
  
  
  public static boolean ApproxEqual(SparseVector x, SparseVector y) {
    boolean res = true;

    if (x.getUsed() == y.getUsed()) {
      double dx[] = x.getData();
      int    ix[] = x.getIndex();
      double dy[] = y.getData();
      int    iy[] = y.getIndex();
      for (int i = 0; i < x.getUsed(); ++i) {
        if (ix[i] != iy[i]) {
          System.out.println(String.format("Different indices detected %d vs %d i=%d", ix[i], iy[i], i));
          res = false;
          break;
        }
        if (Math.abs(dx[i]-dy[i]) > 2* Float.MIN_NORMAL) {
          System.out.println(String.format("Different values detected %f vs %f i=%d index=%d", dx[i], dy[i], i,ix[i]));
          res = false;
          break;
        }
      }
    } else {
      System.err.println(String.format("Different # of elements %d vs %d", x.getUsed(), y.getUsed()));
      res = false;
    }

    if (!res) {
      System.out.println("Different vectors:");
      printSparseVector(x);
      printSparseVector(y);
    }

    return res;
  }
  
  public static synchronized void printSparseVector(SparseVector queryEmbedVector) {
    if (null == queryEmbedVector) return;
    
    double [] data = queryEmbedVector.getData();
    int    [] indx = queryEmbedVector.getIndex();
    
    
    System.out.println("The number of elements: " + queryEmbedVector.getUsed()); 
    
    for (int i = 0; i < queryEmbedVector.getUsed(); ++i) {
      System.out.print(indx[i]+":"+data[i]+" ");
    }
    System.out.println();
  }

  /**
   * Computes a square root of each sparse-vector element. 
   * 
   * @param v
   *      sparse vector
   * @return
   *      a vector of element-wise square roots.
   */
  public static SparseVector sqrt(SparseVector v) {
    SparseVector res = v.copy();
    
    double [] data = res.getData();
    for (int k = 0; k < res.getUsed(); ++k) data[k] = Math.sqrt(data[k]);

    return res;
  }
  
}
