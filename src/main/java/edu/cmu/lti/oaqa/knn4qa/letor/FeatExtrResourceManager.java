/*
 *  Copyright 2018 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.io.File;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.apps.CommonParams;
import edu.cmu.lti.oaqa.knn4qa.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndexFilterAndRecoder;

class Model1Data {
  public final GizaTranTableReaderAndRecoder mRecorder;
  public final float[] mFieldProbTable;
  public Model1Data(GizaTranTableReaderAndRecoder recorder, float[] fieldProbTable) {
    this.mRecorder = recorder;
    this.mFieldProbTable = fieldProbTable;
  }
}

/**
 * This class takes care about loading resources necessary for feature extraction.
 * All resource allocation classes are synchronized on "this" to prevent 
 * race conditions AND DEADLOCKS! Fortunately, Java locks are re-entrant
 * to allow for this.
 * 
 */
public class FeatExtrResourceManager {
  private static final Logger logger = LoggerFactory.getLogger(FeatExtrResourceManager.class);
  
  public static final String FS = File.separator;
  
  private final String mRootModel1TranDir;
  private final String mRootEmbedDir;
  private final String mRootFwdIndexDir;
  
  public FeatExtrResourceManager(String rootFwdIndexDir,
                          String rootModel1TranDir,
                          String rootEmbedDir) {
    mRootFwdIndexDir = rootFwdIndexDir;
    mRootEmbedDir = rootEmbedDir;
    mRootModel1TranDir = rootModel1TranDir;
  }
  
  /**
   * Retrieves and if necessary initializes the forward index:
   * it assumes a standard location and name for forward 
   * 
   * @param fieldName   the name of the field.
   * @return
   * @throws Exception
   */
  public InMemForwardIndex getFwdIndex(String fieldName) throws Exception {
    if (mRootFwdIndexDir == null) {
      throw new Exception("There is no forward index directory, likely, you need to specify " + 
          CommonParams.MEMINDEX_PARAM + " in the calling app");
    }
    // Synchronize all resource allocation on the class reference to avoid race conditions AND dead locks
    synchronized (this) {
      if (!mFwdIndices.containsKey(fieldName)) {
        InMemForwardIndex fwdIndex = new InMemForwardIndex(fwdIndexFileName(mRootFwdIndexDir, fieldName));
        mFwdIndices.put(fieldName, fwdIndex);
      }
      return mFwdIndices.get(fieldName);
    }
  }
  
  public EmbeddingReaderAndRecoder getWordEmbed(String fieldName, String fileName) throws Exception {
    if (mRootEmbedDir == null)
      throw new Exception("There is no forward index directory, likely, you need to specify " + 
          CommonParams.EMBED_DIR_PARAM + " in the calling app");
    // Synchronize all resource allocation on the class reference to avoid race conditions AND dead locks
    synchronized (this) {
      String embedKey = fieldName + "_" + fileName;
      if (!mWordEmbeds.containsKey(embedKey)) {
        InMemForwardIndex fwdIndx = getFwdIndex(fieldName);
        InMemForwardIndexFilterAndRecoder filterAndRecoder = new InMemForwardIndexFilterAndRecoder(fwdIndx);
        mWordEmbeds.put(embedKey, 
            new EmbeddingReaderAndRecoder(mRootEmbedDir +FS + fileName, filterAndRecoder));
      }
      return mWordEmbeds.get(embedKey);
    }
  }
  
  public Model1Data getModel1Tran(String fieldName, 
                                  String model1SubDir, boolean flipTranTable, 
                                  int gizaIterQty, 
                                  float probSelfTran, float minProb) throws Exception {
    if (mRootModel1TranDir == null)
      throw new Exception("There is no forward index directory, likely, you need to specify " + 
          CommonParams.GIZA_ROOT_DIR_PARAM + " in the calling app");
    // Synchronize all resource allocation on the class reference to avoid race conditions AND dead locks
    String key = fieldName + "_" + flipTranTable + "_" + gizaIterQty;
    synchronized (this) {
      if (!mModel1Data.containsKey(key)) {
        InMemForwardIndex fwdIndx = getFwdIndex(fieldName);
        
        InMemForwardIndexFilterAndRecoder filterAndRecoder = new InMemForwardIndexFilterAndRecoder(fwdIndx);
        
        
        String prefix = mRootModel1TranDir + FS + model1SubDir + FS;
        GizaVocabularyReader answVoc  = new GizaVocabularyReader(prefix + "source.vcb", filterAndRecoder);
        GizaVocabularyReader questVoc = new GizaVocabularyReader(prefix + "target.vcb", filterAndRecoder);
    
    
        float[] fieldProbTable = fwdIndx.createProbTable(answVoc);
        
    
        GizaTranTableReaderAndRecoder recorder = new GizaTranTableReaderAndRecoder(
                                         flipTranTable,
                                         prefix + "output.t1." + gizaIterQty,
                                         filterAndRecoder,
                                         answVoc, questVoc,
                                         probSelfTran, 
                                         minProb);
        
        mModel1Data.put(key, new Model1Data(recorder, fieldProbTable));
      }
      
      return mModel1Data.get(key);
    }
  }

  public static String fwdIndexFileName(String prefixDir, String fieldName) {
    return prefixDir + FS + fieldName;
  }  
  
  private HashMap<String, InMemForwardIndex>          mFwdIndices = new HashMap<String, InMemForwardIndex>();
  private HashMap<String, EmbeddingReaderAndRecoder>  mWordEmbeds = new HashMap<String, EmbeddingReaderAndRecoder>();
  private HashMap<String, Model1Data>                 mModel1Data = new HashMap<String, Model1Data>();
}
