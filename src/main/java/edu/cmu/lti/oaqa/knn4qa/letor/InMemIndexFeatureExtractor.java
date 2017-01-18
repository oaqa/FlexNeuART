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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.*;

import javax.annotation.Nullable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Joiner;

import edu.cmu.lti.oaqa.knn4qa.embed.*;
import edu.cmu.lti.oaqa.knn4qa.giza.*;
import edu.cmu.lti.oaqa.knn4qa.memdb.*;
import edu.cmu.lti.oaqa.knn4qa.simil.*;
import net.openhft.koloboke.collect.map.hash.HashIntObjMap;
import net.openhft.koloboke.collect.map.hash.HashIntObjMaps;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.sparse.SparseVector;

/**
 * The base class for the extractor based on the in-memory forward index.
 * 
 * <p>This class actually does all the work. Child classes only override
 * functions like {@code useBM25TextFeature} to specify which features are used.
 * </p>
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class InMemIndexFeatureExtractor extends FeatureExtractor {
  private static final Logger logger = LoggerFactory.getLogger(InMemIndexFeatureExtractor.class);
  
  public static String getExtractorListDesc() {
    Joiner joiner = Joiner.on(',');
    
    String fls [] = {
        InMemIndexFeatureExtractorComplex.CODE,
        InMemIndexFeatureExtractorExper.CODE,
    };
    
    return joiner.join(fls);
  }
  public static InMemIndexFeatureExtractor createExtractor(
      String    extractorType, 
      String    gizaRootDir, 
      int       gizaIterQty,
      String    memIndxPref, 
      @Nullable String   embedDir,
      @Nullable String[] embedFiles,
      @Nullable String[]  highOrderModelFiles) throws Exception {

    if (extractorType.startsWith(InMemIndexFeatureExtractorExper.CODE + "@")) {
     // This is a special extractor that can use parameters specified after the colon
      return 
          new InMemIndexFeatureExtractorExper(
                    extractorType.substring(InMemIndexFeatureExtractorExper.CODE.length() + 1),
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);              
    } else if (extractorType.equals(InMemIndexFeatureExtractorComplex.CODE)) {
      return 
          new InMemIndexFeatureExtractorComplex(
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);
    } else if (extractorType.equals(InMemIndexFeatureExtractorComplexNoMajorEmbed.CODE)) {
      return 
          new InMemIndexFeatureExtractorComplexNoMajorEmbed(
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);
    } else if (extractorType.equals(InMemIndexFeatureExtractorComplexNoJSDComposite.CODE)) {
      return 
          new InMemIndexFeatureExtractorComplexNoJSDComposite(
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);
    } else if (extractorType.equals(InMemIndexFeatureExtractorComplexNoWMD.CODE)) {
      return 
          new InMemIndexFeatureExtractorComplexNoWMD(
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);
    } else if (extractorType.equals(InMemIndexFeatureExtractorComplexNoLCSEmbed.CODE)) {
      return 
          new InMemIndexFeatureExtractorComplexNoLCSEmbed(
                    gizaRootDir, 
                    gizaIterQty, 
                    memIndxPref,
                    embedDir,
                    embedFiles,
                    highOrderModelFiles);
    } 



 

    return null;
  }
  
	/*
	 * There are several non-final constants here. There are several checks
	 * to ensure that the values of these constants are correct. Don't make
	 * constants final and/or private, or Java will mark checking code as dead (this
	 * may result in either a warning or even a compilation error. If you have something as public,
	 * it can be potentially modified by outside code, so the compiler cannot exclude this possibility!  
	 */
	// The number of LCS features for a given field (raw + normalized) for a given field
  public static int LCS_FIELD_FEATURE_QTY = 1; 
  public static int LCS_FIELD_FEATURE_QUERY_NORM_QTY = 1;
  
  // The number of overall-match features (raw + normalized) for a given field
  public static int OVERAL_MATCH_FIELD_FEATURE_QTY = 1;
  public static int OVERAL_MATCH_FIELD_FEATURE_QUERY_NORM_QTY = 1;  
  
  // The number of Model 1 features for a given field (raw + normalized)
  public static int MODEL1_FIELD_FEATURE_QTY = 1;
  public static int MODEL1_FIELD_FEATURE_QUERY_NORM_QTY = 1;
  
  // The number of simple translation features for a given field (raw + fully normalized + query-length normalized)
  public static int SIMPLE_TRAN_FIELD_FEATURE_QTY = 2;
  public static int SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY = 1;
  
  // Jensen-Shannon divergence between composite sparse embeddings
  public static int JSD_COMPOSITE_FEATURE_QTY = 1;
  
  // public The number of BM-25like averaged embedding features per field (we are using only one embedding!)
  static int AVERAGED_EMBEDBM25_FEATURE_QTY = 2;  

  // This is for debugging purposes only, can be used only in a single-thread mode
  public  static boolean PRINT_SCORES                 = false;

  
  public static boolean STRAIGHT_FORWARD_TRAN_COMP   = true;
  
  private static final float LCS_WORD_EMBED_THRESH = 0.5f;
  private static final int   INIT_VOCABULARY_SIZE = 1024*512;
  
  public float getMinModel1Prob(int fieldId) { return mMinModel1ProbDefault[fieldId]; }
  public float getMinSimpleTranProb(int fieldId) { return mMinSimpleTranProbDefault[fieldId]; }
  public float getMinJSDCompositeProb(int fieldId) { return mMinJSDCompositeProbDefault[fieldId]; }
  public float getModel1Lambda(int fieldId) { return mModel1LambdaDefault[fieldId]; }
  public float getProbSelfTran(int fieldId) { return mProbSelfTranDefault[fieldId]; }
  
  /**
   * @return true, if document ranks and scores (as returned by a candidate provider or
   * an intermediate re-ranker) need to be added to the set of features.
   */
  public boolean addRankScores()  { return false; }
  
  /**
   * @return true, if non-weighted average embeddings should be used.
   */
  public boolean useNonWghtAvgEmbed() { return true; }

  /* 
   * use*Feature(int fieldId) functions are defined here.
   * If you add more of these, please, modify the function useField(int fieldId) below.
   */
  public  boolean useBM25Feature(int fieldId) { return false; }
  public  boolean useBM25FeatureQueryNorm(int fieldId) { return false; }
  public  boolean useTFIDFFeature(int fieldId) { return false; }
  public  boolean useTFIDFFeatureQueryNorm(int fieldId) { return false; }
  
  public  boolean useCosineTextFeature(int fieldId) { return false; }
  
  public  boolean useModel1Feature(int fieldId) { return false; }
  public  boolean useModel1FeatureQueryNorm(int fieldId) { return false; }
  public  boolean useSimpleTranFeature(int fieldId) { return false; }
  public  boolean useSimpleTranFeatureQueryNorm(int fieldId) { return false; }
  
  public  boolean useJSDCompositeFeatures(int fieldId) { return false; }
  
  public  boolean useLCSFeature(int fieldId) { return false; }
  public  boolean useLCSFeatureQueryNorm(int fieldId) { return false; }
  public  boolean useOverallMatchFeature(int fieldId) { return false; }
  public  boolean useOverallMatchFeatureQueryNorm(int fieldId) { return false; }
  
  public  boolean useWMDFeatures(int fieldId) { return false; }
  public  boolean useLCSEmbedFeatures(int fieldId) { return false; }
  public  boolean useAveragedEmbedFeatures(int fieldId) { return false; }
  public  boolean useAveragedEmbedBM25Features(int fieldId) { return false; }
  /*
   * End of use*Feature(int fieldId) functions
   */
  boolean useField(int fieldId) {
    return
    useBM25Feature(fieldId) | 
    useBM25FeatureQueryNorm(fieldId) | 
    useTFIDFFeature(fieldId) | 
    useTFIDFFeatureQueryNorm(fieldId) |
    useCosineTextFeature(fieldId) |
    
    useModel1Feature(fieldId) | 
    useModel1FeatureQueryNorm(fieldId) | 
    useSimpleTranFeature(fieldId) | 
    useSimpleTranFeatureQueryNorm(fieldId) | 
    
    useJSDCompositeFeatures(fieldId) | 
    
    useLCSFeature(fieldId) | 
    useLCSFeatureQueryNorm(fieldId) | 
    useOverallMatchFeature(fieldId) | 
    useOverallMatchFeatureQueryNorm(fieldId) | 
    
    useWMDFeatures(fieldId) | 
    useLCSEmbedFeatures(fieldId) | 
    useAveragedEmbedFeatures(fieldId) |
    useAveragedEmbedBM25Features(fieldId);
    
  }
  
  /* Capabilities functions (to check which resources are needed) */
  public boolean needsGIZA() {
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) {
      if (useModel1Feature(fieldId) ||
          useModel1FeatureQueryNorm(fieldId) ||
          useSimpleTranFeature(fieldId) ||
          useSimpleTranFeatureQueryNorm(fieldId)) return true;          
    }
    return false;
  }
  
  public boolean needsSomeEmbed() {
    return needsDenseEmbed() || needsHighOrderEmbed();
  }
  
  public boolean needsDenseEmbed() {
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) {
      if (useWMDFeatures(fieldId) ||
          useLCSEmbedFeatures(fieldId) ||
          useAveragedEmbedFeatures(fieldId) ||
          useAveragedEmbedBM25Features(fieldId)
         ) return true;          
    }
    return false;    
  }
  
  public boolean needsHighOrderEmbed() {
    return useJSDCompositeFeatures(TEXT_FIELD_ID);
  }
  
  /* end of capabilities function */
  

  @Override
  public String getName() {
    return this.getClass().getName();
  }
  
  /**
   * @return    the in-memory index for the text field.
   */
  public InMemForwardIndex getTextFieldIndex() {
    return mFieldIndex[TEXT_FIELD_ID];
  }
  
  /**
   * @return    the in-memory index for a given field field.
   */
  public InMemForwardIndex getFieldIndex(int fieldId) {
    return mFieldIndex[fieldId];
  }
  
  public GizaTranTableReaderAndRecoder getGizaTranTable(int fieldId) {
    return maAnswToQuestTran[fieldId];
  }
  
  public QueryDocSimilarity getBM25NormSimil(int fieldId) {
    return mBM25SimilarityNorm[fieldId];
  }
  
  public EmbeddingReaderAndRecoder getEmbeddingReaderAndRecoder(int fieldId, int k) {
    return mWordEmbeds[fieldId][k];
  }
    
  /**
   * Constructor, which doesn't really initialize: a separate function will do the real initialization;
   * this is done on purpose: an uninitialized instance can be used to check capabilities.
   * 
   * 
   * @param dirTranPrefix
   *            a prefix of the directories containing translation table.
   * @param gizaIterQty
   *            the number of GIZA++ iteration.
   * @param indexDir
   *            the directory that keeps serialized in-memory index.
   * @param embedDir
   *            the directory that keeps word embeddings.
   * @param embedFiles
   *            the list of dense word embedding files (relative to embedDir)
   * @param highOrderModelFiles
   *            the list of sparse word embedding files based on translation probabilities (relative to embedDir)
   *                       
   * @throws Exception
   */
  public InMemIndexFeatureExtractor(String dirTranPrefix, 
                                    int gizaIterQty, 
                                    String indexDir,
                                    @Nullable String    embedDir,
                                    @Nullable String[]  embedFiles,
                                    @Nullable String[]  highOrderModelFiles) {
    mDirTranPrefix       = dirTranPrefix;
    mGizaIterQty         = gizaIterQty;
    mIndexDir            = indexDir;
    mEmbedDir            = embedDir;
    mEmbedFiles          = embedFiles;
    mHighOrderModelFiles = highOrderModelFiles;
    
    mHighOrderModels = new ArrayList<ArrayList<HashIntObjMap<SparseVector>>>();
    for (int fieldId = 0; fieldId < mFieldIndex.length; ++fieldId) {
      mHighOrderModels.add(null);
    }
  }
    
  /* LIMITATION: initFieldIndex should be called in the order of increasing fieldId! */
  void initFieldIndex(int fieldId, InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    // First try to reuse donor's index
    for (int donorId = 0; donorId < donorExtractors.length; donorId++) {
      InMemIndexFeatureExtractor donnor = donorExtractors[donorId];
      if (null == donnor) continue;   
      if (null == mFieldIndex[fieldId]) 
        mFieldIndex[fieldId] = donnor.mFieldIndex[fieldId];     
    }
    /*
     *  If a donor doesn't have one, create a new one from scratch,
     *  unless we have an alias.
     */
    if (null == mFieldIndex[fieldId]) {
      int aliasOfId = FeatureExtractor.mAliasOfId[fieldId];
      // First, let's do a couple of paranoid checks
      if (aliasOfId >= 0) {
        if (aliasOfId >= fieldId) {
          throw 
            new RuntimeException("Bug: the id of the alias " + fieldId + 
                                 " is smaller than the id of the field it mirrors!");
        }
        if (null == mFieldIndex[aliasOfId]) {
          logger.info("Field " + FeatureExtractor.mFieldNames[fieldId] +
                      " : the field index of the alias " + FeatureExtractor.mFieldNames[aliasOfId] + 
              " is not initialized, initting the index from scratch!");
          // Note that the index is initialized using the name of the aliased field!
          mFieldIndex[fieldId] = new InMemForwardIndex(indexFileName(mIndexDir, FeatureExtractor.mFieldNames[aliasOfId]));
        } else {
          // All is fine, we can reuse the field index of the aliased field
          logger.info("Field " + FeatureExtractor.mFieldNames[fieldId] +
              " : the field index of the alias " + FeatureExtractor.mFieldNames[aliasOfId] + 
      " is already initialized, so we reuse this index.");
          mFieldIndex[fieldId] = mFieldIndex[aliasOfId];
        }
      } else 
        mFieldIndex[fieldId] = new InMemForwardIndex(indexFileName(mIndexDir, FeatureExtractor.mFieldNames[fieldId]));
    }
  }
  
  void initHighorderModels(int fieldId, InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    // First try to reuse donor's models
    for (int donorId = 0; donorId < donorExtractors.length; donorId++) {
      InMemIndexFeatureExtractor donnor = donorExtractors[donorId];
      if (null == donnor) continue;
      if (null == mHighOrderModels.get(fieldId)) 
        mHighOrderModels.set(fieldId, donnor.mHighOrderModels.get(fieldId));
    }
    // If a donor doesn't have models, create new models from scratch
    if (mHighOrderModels.get(fieldId) == null) {
      initFieldIndex(fieldId, donorExtractors);            
      
      mHighOrderModels.set(fieldId, new ArrayList<HashIntObjMap<SparseVector>>());          
      
      for (int k = 0; k < mHighOrderModelFiles.length; ++k) {
        String fileName = mHighOrderModelFiles[k];
        mHighOrderModels.get(fieldId).add(SparseEmbeddingReaderAndRecorder.readDict(mFieldIndex[fieldId], mEmbedDir + "/" + fileName));
        logger.info("Read ebmedding file: " + fileName);
      }
    }

  }
  
  void initAnswToQuestTran(int fieldId, InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    // First try to reuse donor's translation models
    for (int donorId = 0; donorId < donorExtractors.length; donorId++) {
      InMemIndexFeatureExtractor donnor = donorExtractors[donorId];
      if (null == donnor) continue;
      if (null == maAnswToQuestTran[fieldId]) { 
        maAnswToQuestTran[fieldId] = donnor.maAnswToQuestTran[fieldId];
        maFieldProbTable[fieldId] = donnor.maFieldProbTable[fieldId];
      }
    }
    if (null == maAnswToQuestTran[fieldId]) {
      initFieldIndex(fieldId, donorExtractors);
      // If reuse fails, create new ones
      InMemForwardIndexFilterAndRecoder filterAndRecoder = new InMemForwardIndexFilterAndRecoder(mFieldIndex[fieldId]);
  
      
      String prefix = mDirTranPrefix + "/" + FeatureExtractor.mFieldNames[fieldId] + "/";
      GizaVocabularyReader answVoc  = new GizaVocabularyReader(prefix + "source.vcb", filterAndRecoder);
      GizaVocabularyReader questVoc = new GizaVocabularyReader(prefix + "target.vcb", filterAndRecoder);
  
  
      maFieldProbTable[fieldId] = mFieldIndex[fieldId].createProbTable(answVoc);
      
  
      maAnswToQuestTran[fieldId] = new GizaTranTableReaderAndRecoder(
                                       mFlippedTranTableFieldUse[fieldId],
                                       prefix + "/output.t1." + mGizaIterQty,
                                       filterAndRecoder,
                                       answVoc, questVoc,
                                       (float)getProbSelfTran(fieldId), 
                                       (float)Math.min(getMinModel1Prob(fieldId), 
                                                       Math.min(getMinSimpleTranProb(fieldId), getMinJSDCompositeProb(fieldId)))
                                                       );
    }
  }  
  
  void initWordEmbeds(int fieldId, InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    // First try to reuse donor's embeddings
    for (int donorId = 0; donorId < donorExtractors.length; donorId++) {
      InMemIndexFeatureExtractor donnor = donorExtractors[donorId];
      if (null == donnor) continue;
      if (null == mWordEmbeds[fieldId])
        mWordEmbeds[fieldId] = donnor.mWordEmbeds[fieldId];     
    }
    // If reuse fails, create new ones
    if (mWordEmbeds[fieldId] == null) {
      int embedQty = mEmbedFiles.length;
      
      mWordEmbeds[fieldId] = new EmbeddingReaderAndRecoder[embedQty];
      
      for (int i = 0; i < embedQty; ++i) {
        InMemForwardIndexFilterAndRecoder filterAndRecoder = new InMemForwardIndexFilterAndRecoder(mFieldIndex[fieldId]);

        mWordEmbeds[fieldId][i] = new EmbeddingReaderAndRecoder(mEmbedDir + "/" + mEmbedFiles[i], filterAndRecoder);
      }
    }
  }  
  
  /**
   * An actual initialization function, which can "borrow" some the resources (field indices, embeddings,
   * translation tables, etc from another extractor).
   * 
   * @throws Exception
   */
  public void init(InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    logger.info(String.format("(if averaged embeddings are used at all) using non-weighted average embeddings=%b", useNonWghtAvgEmbed()));
    
    /*
     * First, let's do several paranoid checks to ensure that the number of elements in all
     * arrays describing fields is the same. As a code improvement, it makes sense to 
     * extract all such arrays into a single class.
     */
    {
      int fieldQty = FeatureExtractor.mFieldNames.length;
      if (FeatureExtractor.mFieldsSOLR.length != fieldQty)
        throw new RuntimeException("Bug: FeatureExtractor.mFieldsSOLR.length != fieldQty");
      
      if (FeatureExtractor.mAliasOfId.length != fieldQty)
        throw new RuntimeException("Bug: FeatureExtractor.mAliasOfId.length != fieldQty");
      
      if (mFlippedTranTableFieldUse.length != fieldQty)
        throw new RuntimeException("Bug: mFlippedTranTableFieldUse.length != fieldQty");
      
      if (mModel1LambdaDefault.length != fieldQty)
        throw new RuntimeException("Bug: mModel1LambdaDefault.length != fieldQty");
      
      if (mProbSelfTranDefault.length != fieldQty)
        throw new RuntimeException("Bug: mProbSelfTranDefault.length != fieldQty");
      
      if (mMinModel1ProbDefault.length != fieldQty)
        throw new RuntimeException("Bug: mMinModel1ProbDefault.length != fieldQty");
      
      if (mMinJSDCompositeProbDefault.length != fieldQty)
        throw new RuntimeException("Bug: mMinJSDCompositeProbDefault.length != fieldQty");
      
      if (mMinSimpleTranProbDefault.length != fieldQty)
        throw new RuntimeException("Bug: mMinSimpleTranProbDefault.length != fieldQty");
    }
    
    {            
            
  		int qty = 0;
	    for (int fieldId = 0; fieldId < mFieldIndex.length; ++fieldId) {
	      logger.info(String.format("field=%s useBM25Feature=%b useTFIDFFeature=%b useBM25FeatureQueryNorm=%b useTFIDFFeatureQueryNorm=%b useCosineTextFeature=%b",
	          mFieldNames[fieldId],
	          useBM25Feature(fieldId), useTFIDFFeature(fieldId), useBM25FeatureQueryNorm(fieldId), useTFIDFFeatureQueryNorm(fieldId), useCosineTextFeature(fieldId)));
	      
		    if (useBM25Feature(fieldId) || useTFIDFFeature(fieldId) || useBM25FeatureQueryNorm(fieldId) || useTFIDFFeatureQueryNorm(fieldId) || useCosineTextFeature(fieldId) ||
		       // Word embedding based features will be used only for text-fields
		       (
		        (useWMDFeatures(fieldId) || useLCSEmbedFeatures(fieldId) || useAveragedEmbedFeatures(fieldId) || useAveragedEmbedBM25Features(fieldId)) 
		        && isSomeTextFieldId(fieldId)
		        ) 
		       ) {
		      initFieldIndex(fieldId, donorExtractors);
		        
	        mBM25Similarity[fieldId]        = new BM25SimilarityLucene(BM25_K1, BM25_B, mFieldIndex[fieldId]);
	        mDefaultSimilarity[fieldId]     = new DefaultSimilarityLucene(mFieldIndex[fieldId]);
          mBM25SimilarityNorm[fieldId]    = new BM25SimilarityLuceneNorm(BM25_K1, BM25_B, mFieldIndex[fieldId]);
          mDefaultSimilarityNorm[fieldId] = new DefaultSimilarityLuceneNorm(mFieldIndex[fieldId]);
          mCosineTextSimilarity[fieldId]  = new CosineTextSimilarity(mFieldIndex[fieldId]);
		    }
	        
        if (useBM25Feature(fieldId))      ++qty;
        if (useTFIDFFeature(fieldId))     ++qty;
        if (useBM25FeatureQueryNorm(fieldId))  ++qty;
        if (useTFIDFFeatureQueryNorm(fieldId)) ++qty;
        if (useCosineTextFeature(fieldId)) ++qty;
	    }	    
	    mFieldScoreFeatQty = qty;
  	}
    
  	{
  	  int qty = 0;

  	  for (int fieldId = 0; fieldId < mFieldNames.length; ++fieldId) { 
  	    logger.info(String.format("field=%s useLCSFeature=%b useLCSFeatureQueryNorm=%b",
            mFieldNames[fieldId], useLCSFeature(fieldId), useLCSFeatureQueryNorm(fieldId)));
  	    if (useLCSFeature(fieldId) || useLCSFeatureQueryNorm(fieldId)) {
  	      initFieldIndex(fieldId, donorExtractors);
  	    }
  	    if (useLCSFeature(fieldId))          qty += LCS_FIELD_FEATURE_QTY; 
  	    if (useLCSFeatureQueryNorm(fieldId)) qty += LCS_FIELD_FEATURE_QUERY_NORM_QTY;
  	  }
  	  mLCSFeatQty = qty;
  	}
    
  	{
  	  int qty = 0;
  	  for (int fieldId = 0; fieldId < mFieldNames.length; ++fieldId) {
  	    logger.info(String.format("field=%s useOverallMatchFeature=%b useOverallMatchFeatureQueryNorm=%b",
            mFieldNames[fieldId], useOverallMatchFeature(fieldId), useOverallMatchFeatureQueryNorm(fieldId)));
  	    
  	    if (useOverallMatchFeature(fieldId) || useOverallMatchFeatureQueryNorm(fieldId)) {
  	      initFieldIndex(fieldId, donorExtractors);
  	    }  	    
  	    if (useOverallMatchFeature(fieldId))          qty += OVERAL_MATCH_FIELD_FEATURE_QTY;
  	    if (useOverallMatchFeatureQueryNorm(fieldId)) qty += OVERAL_MATCH_FIELD_FEATURE_QUERY_NORM_QTY;  	    
  	  }
  	  mOverallMatchFeatQty = qty;
  	}
    
    {
      int qty_model1_tran = 0, qty_simple_tran = 0;
      int qty_jsd_comp = 0;
      
      if (FeatureExtractor.mFieldNames.length != mFieldIndex.length) {
      	throw new RuntimeException("Bug: the number of translation directions isn't the same as the number of fields!");
      }
      

      for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) {
        boolean useModel1         = useModel1Feature(fieldId);
        boolean useSimpleTran     = useSimpleTranFeature(fieldId);
        boolean useJSDComp        = useJSDCompositeFeatures(fieldId);
        boolean useModel1Norm     = useModel1FeatureQueryNorm(fieldId);
        boolean useSimpleTranNorm = useSimpleTranFeatureQueryNorm(fieldId);
        
        logger.info(String.format("field=%s useModel1Feature=%b useModel1FeatureQueryNorm=%b minProbModel1=%g probSelfTran=%g lambdaModel1=%g",
            mFieldNames[fieldId], useModel1, useModel1Norm, getMinModel1Prob(fieldId), getProbSelfTran(fieldId), getModel1Lambda(fieldId)));
        logger.info(String.format("field=%s useSimpleTranFeature=%b useSimpleTranFeatureQueryNorm=%b minProbSimpleTran=%g probSelfTran=%g",
            mFieldNames[fieldId], useSimpleTran, useSimpleTranNorm, getMinSimpleTranProb(fieldId), getProbSelfTran(fieldId)));
        logger.info(String.format("field=%s useJSDComposite=%b", mFieldNames[fieldId], useJSDComp));
        
  			
        if (useJSDComp) { 
          if (!isSomeTextFieldId(fieldId)) {
            throw new Exception("JSD-composite word embeddings should only be used with a text field");
          }
          
          initHighorderModels(fieldId, donorExtractors);            

          qty_jsd_comp += JSD_COMPOSITE_FEATURE_QTY * mHighOrderModelFiles.length;
          
            
        }
        
  			if (useModel1 || useSimpleTran || useModel1Norm || useSimpleTranNorm) {
          if (useModel1)           qty_model1_tran += MODEL1_FIELD_FEATURE_QTY;
          if (useModel1Norm)       qty_model1_tran += MODEL1_FIELD_FEATURE_QUERY_NORM_QTY;
          if (useSimpleTran)       qty_simple_tran += SIMPLE_TRAN_FIELD_FEATURE_QTY;
          if (useSimpleTranNorm)   qty_simple_tran += SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY;

          
          initFieldIndex(fieldId, donorExtractors);
          initAnswToQuestTran(fieldId, donorExtractors);                                       
  			}
      }
           
      mModel1FeatQty     = qty_model1_tran;
      mSimpleTranFeatQty = qty_simple_tran;     
      mJSDCompositeFeatQty= qty_jsd_comp;
    }     

    {
      int qty_averaged_embed = 0, qty_averaged_embedbm25 = 0, qty_wmd = 0, qty_lcs_embed = 0;

      for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) {
        logger.info(String.format("field=%s useWMDFeatures=%b useLCSEmbedFeatures=%b useAveragedEmbedFeatures=%b useAveragedEmbedBM25Features",
            mFieldNames[fieldId], useWMDFeatures(fieldId), useLCSEmbedFeatures(fieldId), useAveragedEmbedFeatures(fieldId), useAveragedEmbedBM25Features(fieldId)));
        
        if ((useWMDFeatures(fieldId) || useLCSEmbedFeatures(fieldId) || useAveragedEmbedFeatures(fieldId) || useAveragedEmbedBM25Features(fieldId))) {
          if (!isSomeTextFieldId(fieldId)) {
            throw new Exception("Dense word embeddings can only be used with text fields (lemmatized and original)!");
          }
          
          initFieldIndex(fieldId, donorExtractors);                   
    
          if (null == mEmbedFiles)
            throw new Exception("Expecting a non-null list of embedding files!");
          if (null == mEmbedDir)
            throw new Exception("Expecting a non-null embedding directory!");
          
          initWordEmbeds(fieldId, donorExtractors);
          
          int embedQty = mEmbedFiles.length;
          
          if (useAveragedEmbedFeatures(fieldId)) qty_averaged_embed += (1 + (useNonWghtAvgEmbed() ? 1 : 0)) * embedQty;
          if (useAveragedEmbedBM25Features(fieldId)) qty_averaged_embedbm25 += AVERAGED_EMBEDBM25_FEATURE_QTY;
          if (useWMDFeatures(fieldId))           qty_wmd            += DistanceFunctions.EMD_LIKE_QTY; // Uses only one embedding file
          if (useLCSEmbedFeatures(fieldId))      qty_lcs_embed      += 2*DistanceFunctions.LCS_LIKE_QTY; // 2 because regular + normalized, alsoe uses one embedding file
        }
      }
      mAveragedEmbedFeatureQty = qty_averaged_embed;
      mAveragedEmbedBM25FeatureQty = qty_averaged_embedbm25;
      mLCSEmbedFeatQty         = qty_lcs_embed;
      mWMDFeatQty              = qty_wmd;
    }
  }  
  
  @Override
  public Map<String, DenseVector> getFeatures(ArrayList<String> arrDocIds,
                                               Map<String, String> queryData) throws Exception 
  {
    HashMap<String,DenseVector> res = new HashMap<String,DenseVector>();
    
    int featureQty = getFeatureQty();
    for (String docId : arrDocIds) {
      res.put(docId, new DenseVector(featureQty));
    }    
    
    int id = 0;
    

    for (int fieldId = 0; fieldId < mFieldsSOLR.length; ++fieldId) {       
			String query = queryData.get(mFieldsSOLR[fieldId]);
	    String fieldName = FeatureExtractor.mFieldNames[fieldId];

      if (useBM25Feature(fieldId)) {
	      getFieldScores(mFieldIndex[fieldId], mBM25Similarity[fieldId],
	                     arrDocIds, fieldName,   
	                     id++, query, res);
      }
      
      if (useTFIDFFeature(fieldId)) {
	      getFieldScores(mFieldIndex[fieldId], mDefaultSimilarity[fieldId],
	                    arrDocIds, fieldName,   
	                    id++, query, res);
      }
      
      if (useCosineTextFeature(fieldId)) {
        getFieldScores(mFieldIndex[fieldId], mCosineTextSimilarity[fieldId],
                      arrDocIds, fieldName,   
                      id++, query, res);
      }      
      
      if (useBM25FeatureQueryNorm(fieldId)) {
        getFieldScores(mFieldIndex[fieldId], mBM25SimilarityNorm[fieldId],
                       arrDocIds, fieldName,   
                       id++, query, res);
      }
      
      if (useTFIDFFeatureQueryNorm(fieldId)) {
        getFieldScores(mFieldIndex[fieldId], mDefaultSimilarityNorm[fieldId],
                      arrDocIds, fieldName,   
                      id++, query, res);
      }
      
      if (useOverallMatchFeature(fieldId) || useOverallMatchFeatureQueryNorm(fieldId)) {
        getFieldOverallMatchScores(mFieldIndex[fieldId], fieldId,
                                  arrDocIds, fieldName, 
                                  id, query, res);
        if (useOverallMatchFeature(fieldId))           id += OVERAL_MATCH_FIELD_FEATURE_QTY;
        if (useOverallMatchFeatureQueryNorm(fieldId))  id += OVERAL_MATCH_FIELD_FEATURE_QUERY_NORM_QTY;
      }     
  
      if (useLCSFeature(fieldId) || useLCSFeatureQueryNorm(fieldId)) {
        getFieldLCSScores(mFieldIndex[fieldId], fieldId,
                          arrDocIds, fieldName, 
                          id, query, res);
                
        if (useLCSFeature(fieldId))           id += LCS_FIELD_FEATURE_QTY;
        if (useLCSFeatureQueryNorm(fieldId))  id += LCS_FIELD_FEATURE_QUERY_NORM_QTY;
      }
	    
			boolean useModel1         = useModel1Feature(fieldId);
			boolean useSimpleTran     = useSimpleTranFeature(fieldId);

      boolean useModel1Norm     = useModel1FeatureQueryNorm(fieldId);
	    boolean useSimpleTranNorm = useSimpleTranFeatureQueryNorm(fieldId);

			
      if (useModel1 || useSimpleTran || useModel1Norm || useSimpleTranNorm) {
  	  
        if (useModel1 || useSimpleTran || useModel1Norm || useSimpleTranNorm) {
  				if (mFlippedTranTableFieldUse[fieldId]) {
  				  getFieldAllTranScoresFlipped(
  				      mFieldIndex[fieldId],
  				      fieldId,
  				      maFieldProbTable[fieldId],
  				      arrDocIds, 
  				      fieldName, 
  				      getMinModel1Prob(fieldId), 
  				      getMinSimpleTranProb(fieldId),
  				      id, query, 
  				      maAnswToQuestTran[fieldId],
  				      getModel1Lambda(fieldId), OOV_PROB,
  				      res);
  				} else {
  				  getFieldAllTranScoresDirect(
  				      mFieldIndex[fieldId], 
  				      fieldId,
  				      maFieldProbTable[fieldId],
  				      arrDocIds, 
  				      fieldName, 
  				      getMinModel1Prob(fieldId), 
  				      getMinSimpleTranProb(fieldId),
  				      id, query, 
  				      maAnswToQuestTran[fieldId],
  				      getModel1Lambda(fieldId), OOV_PROB,
  				      res);
  				}
        }
				

				  			
	      if (useModel1)
	        id += MODEL1_FIELD_FEATURE_QTY;
        if (useModel1Norm)
          id += MODEL1_FIELD_FEATURE_QUERY_NORM_QTY;	      
	      if (useSimpleTran)
	        id += SIMPLE_TRAN_FIELD_FEATURE_QTY;
        if (useSimpleTranNorm)
          id += SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY;	          
      }        
      
      if (useJSDCompositeFeatures(fieldId)) {
          getFieldJSDCompositeScores(
              mFieldIndex[fieldId], 
              fieldId,
              maFieldProbTable[fieldId],
              arrDocIds, 
              fieldName,
              getMinJSDCompositeProb(fieldId),
              id, query, 
              maAnswToQuestTran[fieldId],              
              res);
          id += mHighOrderModels.get(fieldId).size() * JSD_COMPOSITE_FEATURE_QTY;          
      }			
 	  
      // Embeddings are used only for textual fields
      if ((useWMDFeatures(fieldId) || useLCSEmbedFeatures(fieldId) || useAveragedEmbedFeatures(fieldId) || useAveragedEmbedBM25Features(fieldId)) && 
          isSomeTextFieldId(fieldId)) {
        if (null == mWordEmbeds[fieldId]) {
          throw new Exception("Bug: no embeddings for fieldId="+fieldId);
        }
        
        getFieldEmbedScores(mFieldIndex[fieldId],
                          fieldId,
                          mBM25Similarity[fieldId],
                          arrDocIds, fieldName,
                          id, query, res);
        
        if (useWMDFeatures(fieldId))
          id += DistanceFunctions.EMD_LIKE_QTY;
        if (useLCSEmbedFeatures(fieldId))
          id += 2 * DistanceFunctions.LCS_LIKE_QTY; // 2, because we use both original and normalized
        if (useAveragedEmbedFeatures(fieldId))
          id += (1 + (useNonWghtAvgEmbed() ? 1 : 0)) * mWordEmbeds[fieldId].length;
        if (useAveragedEmbedBM25Features(fieldId))
          id += AVERAGED_EMBEDBM25_FEATURE_QTY;
      }
    }
    
    if (id != getFeatureQty()) {
    	throw 
    	new RuntimeException(
    			String.format("Bug: expected to create %d features, but actually created %d", 
    											getFeatureQty(), id));
    }    
         
    return res;
  }

  @Override
  public int getFeatureQty() {
    return getFieldScoreFeatQty() +
           getModel1FeatQty() +
           getSimpleTranFeatQty() +
           getJSDCompositeFeatQty() +      
           getLCSFeatQty() +
           getOverallMatchFeatQty() +
           getWMDFeatQty() + getLCSEmbedFeatQty() + getAvgEmbedFeatQty() + getAvgEmbedBM25FeatQty();
  }
  
  public int getFieldScoreFeatQty() {
    return mFieldScoreFeatQty;  
  }

  public int getModel1FeatQty() {
    return mModel1FeatQty;
  }
  
  public int getSimpleTranFeatQty() {
    return mSimpleTranFeatQty;
  }

  public int getJSDCompositeFeatQty() {
    return mJSDCompositeFeatQty;
  }
  
  public int getLCSFeatQty() {
    return mLCSFeatQty;
  }
  
  public int getOverallMatchFeatQty() {
    return mOverallMatchFeatQty;
  }
  
  public int getWMDFeatQty() {
    return mWMDFeatQty;
  }
  
  public int getLCSEmbedFeatQty() {
    return mLCSEmbedFeatQty;
  }
  
  public int getAvgEmbedFeatQty() {
    return mAveragedEmbedFeatureQty; 
  }
    
  public int getAvgEmbedBM25FeatQty() {
    return mAveragedEmbedBM25FeatureQty; 
  }  
  /**
   * Get overall match scores for one field.
   * 
   * @param fieldIndex      an in-memory field index
   * @param fieldId         a field identifier
   * @param arrDocIds       an array of document ids.
   * @param fieldName       a name of the field. 
   * @param startFeatureId  an index/id of the first feature.
   * @param query           an actual query
   * @param res             a result set to be updated.   * 
   * @throws Exception
   */
  private void getFieldOverallMatchScores(InMemForwardIndex fieldIndex, int fieldId,
                          ArrayList<String> arrDocIds, 
                          String fieldName,
                          int startFeatureId,
                          String query,
                          Map<String,DenseVector> res) throws Exception {    
    if (null == query) return;
    query = query.trim();
    if (query.isEmpty()) return;
    
    DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
    
    
    if (PRINT_SCORES)
      System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldOverallMatchScores))");
   
    for (String docId : arrDocIds) {
      DocEntry docEntry = fieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      float score = DistanceFunctions.compOverallMatch(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }

      float scoreQueryNorm = score / Math.max(1, queryEntry.mWordIds.length);
      
      int fid = startFeatureId;
      if (useOverallMatchFeature(fieldId)) {
        if (OVERAL_MATCH_FIELD_FEATURE_QTY != 1) {
          throw new RuntimeException("Wrong value of constant OVERAL_MATCH_FIELD_FEATURE_QTY");
        }
        v.set(fid++, score);
      }
      if (useOverallMatchFeatureQueryNorm(fieldId)) {
        if (OVERAL_MATCH_FIELD_FEATURE_QUERY_NORM_QTY!= 1) {
          throw new RuntimeException("Wrong value of constant OVERAL_MATCH_FIELD_FEATURE_QUERY_NORM_QTY");
        }
        v.set(fid++, scoreQueryNorm);
      }
      
      
      if (PRINT_SCORES) {
        if (useOverallMatchFeature(fieldId))
          System.out.println(String.format("Doc id %s %s: %g", docId, "OVERALL MATCH", score));
        if (useOverallMatchFeatureQueryNorm(fieldId))
          System.out.println(String.format("Doc id %s %s: %g", docId, "OVERALL MATCH QUERY-NORM", scoreQueryNorm));        
      }
    }      
  }

  
/**
 * Get LCS scores for one field.
 * 
 * @param fieldIndex      an in-memory field index
 * @param fieldId         a field identifier
 * @param arrDocIds       an array of document ids.
 * @param fieldName       a name of the field. 
 * @param startFeatureId  an index/id of the first feature.
 * @param query           an actual query
 * @param res             a result set to be updated.   * 
 * @throws Exception
 */
private void getFieldLCSScores(InMemForwardIndex fieldIndex, int fieldId,
                        ArrayList<String> arrDocIds, 
                        String fieldName,
                        int startFeatureId,
                        String query,
                        Map<String,DenseVector> res) throws Exception {    
  if (null == query) return;
  query = query.trim();
  if (query.isEmpty()) return;
  
  DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
  
  boolean useLCSFeature          = useLCSFeature(fieldId);
  boolean useLCSFeatureQueryNorm = useLCSFeatureQueryNorm(fieldId);
  
  if (PRINT_SCORES)
    System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldLCSScores))");
 
  for (String docId : arrDocIds) {
    DocEntry docEntry = fieldIndex.getDocEntry(docId);
    
    if (docEntry == null) {
      throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
    }
    
    float score = DistanceFunctions.compLCS(queryEntry.mWordIdSeq, docEntry.mWordIdSeq);
    
    DenseVector v = res.get(docId);
    if (v == null) {
      throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
    }

    float normScore = score / Math.max(1, queryEntry.mWordIdSeq.length);
    
    int fid = startFeatureId;
    
    if (useLCSFeature) {
      v.set(fid++, score);      
      if (LCS_FIELD_FEATURE_QTY != 1) {
        throw new RuntimeException("Bug: wrong value for the constant LCS_FIELD_FEATURE_QTY");
      }
    }
    if (useLCSFeatureQueryNorm) {
      v.set(fid++, normScore);
      if (LCS_FIELD_FEATURE_QUERY_NORM_QTY != 1) {
        throw new RuntimeException("Bug: wrong value for the constant LCS_FIELD_FEATURE_QUERY_NORM_QTY");
      }      
    }
    
    
    if (PRINT_SCORES) {
      if (useLCSFeatureQueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "LCS", score));
      if (useLCSFeatureQueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "LCS (query-norm)", normScore));
    }
  }
  
 }

/**
 * Get all (IBM Model1 + simple) translation scores for one field.
 * 
 * <p>It is designed to worked only for the direct translation table!</p>
 * 
 * @param fieldIndex        an in-memory field index
 * @param fieldId           a field identifier
 * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
 * @param arrDocIds         an array of document ids.
 * @param fieldName         a name of the field. 
 * @param minModel1Prob     a minimum Model1 probability for the field.
 * @param minSimpleTranProb a minimum simple tran. probability for the field
 * @param startFeatureId    an index/id of the first feature.
 * @param query             an actual query
 * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++) 
 * @param lambda            smoothing coefficient
 * @param outOfVocProb      a probability for the out-of-vocabulary word 
 * @param res               a result set to be updated.    
 * @throws Exception
 */
private void getFieldAllTranScoresDirect(InMemForwardIndex fieldIndex,
											  int fieldId,
                        float[] fieldProbTable,
                        ArrayList<String> arrDocIds, 
                        String fieldName,
                        float minModel1Prob, float minSimpleTranProb, 
                        int startFeatureId,
                        String query,
                        GizaTranTableReaderAndRecoder answToQuestTran,
                        double lambda, 
                        double outOfVocProb, 
                        Map<String,DenseVector> res) throws Exception {    
  if (null == query) return;
  query = query.trim();
  if (query.isEmpty()) return;
  
  final float PROB_SELF_TRAN = getProbSelfTran(fieldId); 
  
  boolean useModel1         = useModel1Feature(fieldId);
  boolean useSimpleTran     = useSimpleTranFeature(fieldId);
  boolean useModel1QueryNorm     = useModel1FeatureQueryNorm(fieldId);
  boolean useSimpleTranQueryNorm = useSimpleTranFeatureQueryNorm(fieldId);
  
  DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
    
  int queryWordQty = queryEntry.mWordIds.length;
  
  float queryNorm = Math.max(1, queryWordQty);
    
  if (PRINT_SCORES)
    System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldAllTranScoresDirect))");
 
  for (String docId : arrDocIds) {
    DocEntry docEntry = fieldIndex.getDocEntry(docId);
    
    if (docEntry == null) {
      throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
    }
    
    double logScore = 0;
    float  shareTranPairQty = 0;
    
    float [] aSourceWordProb = new float[docEntry.mWordIds.length];        
    float sum = 0;    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) 
      sum += docEntry.mQtys[ia];
    float invSum = 1/Math.max(1, sum);
      
    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) {
      /*
      int answWordId = docEntry.mWordIds[ia];
      aSourceWordProb[ia] = fieldProbTable[answWordId];
      */
      aSourceWordProb[ia] = docEntry.mQtys[ia] * invSum;
    }

    if (STRAIGHT_FORWARD_TRAN_COMP) {
      /*
       *  Despite potentially worse locality of reference and 
       *  using many more hash lookups, this code apparently
       *  has the same run-time as a more sophisticated version.
       *  
       */
      for (int iq=0; iq < queryEntry.mWordIds.length;++iq) {
        float totTranProb = 0;
        
        int queryWordId = queryEntry.mWordIds[iq];
        int queryRepQty    = queryEntry.mQtys[iq];
        
        if (queryWordId >= 0) {          
          for (int ia = 0; ia < docEntry.mWordIds.length; ++ia) {
            int answWordId = docEntry.mWordIds[ia];
            int answRepQty = docEntry.mQtys[ia];
            
            float oneTranProb = answToQuestTran.getTranProb(answWordId, queryWordId);
            if (answWordId == queryWordId && PROB_SELF_TRAN - oneTranProb > Float.MIN_NORMAL) {
              System.err.println("No self-tran probability for: id=" + answWordId + "!");
              System.exit(1);
            }                
            if (oneTranProb >= minModel1Prob) {
              totTranProb += oneTranProb * aSourceWordProb[ia];
            }
            if (oneTranProb >= minSimpleTranProb) {
              shareTranPairQty += answRepQty * queryRepQty;
            }
          }
        }
  
        double collectProb = queryWordId >= 0 ? Math.max(outOfVocProb, fieldProbTable[queryWordId]) : outOfVocProb;
                                                    //answToQuestTran.getSourceWordProb(queryWordId)); 
        
        logScore += Math.log((1-lambda)*totTranProb +lambda*collectProb);
      }
    } else {
      float [] totTranProb = new float[queryEntry.mWordIds.length];

      for (int iaOuterLoop = 0; iaOuterLoop < docEntry.mWordIds.length; ++iaOuterLoop) {
        int answRepQty = docEntry.mQtys[iaOuterLoop];
        int answWordIdOuterLoop = docEntry.mWordIds[iaOuterLoop];
      
        GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(answWordIdOuterLoop);
        
        if (null != tranRecs) {
          int iq = 0;
          int ia = 0;
          
          while (iq < queryEntry.mWordIds.length && ia < tranRecs.mDstIds.length) {
            int queryWordId = queryEntry.mWordIds[iq];
            int queryRepQty = queryEntry.mQtys[iq];
            int answTranWordId = tranRecs.mDstIds[ia];
            
            if (queryWordId < answTranWordId)
              iq++;
            else if (queryWordId > answTranWordId)
              ia++;
            else {              
              if (answTranWordId >= 0) { // ignore out-of voc words
                float oneTranProb = tranRecs.mProbs[ia];
                
                if (answWordIdOuterLoop == queryWordId && PROB_SELF_TRAN - oneTranProb > Float.MIN_NORMAL) {
                  System.err.println(
                      String.format("No self-tran probability for: id=%d tran prob=%g", 
                        answTranWordId, oneTranProb));
                  System.exit(1);
                }                
                if (oneTranProb >= minModel1Prob) {
                  totTranProb[iq] += oneTranProb  * aSourceWordProb[iaOuterLoop];
                }
                if (oneTranProb >= minSimpleTranProb) {
                  shareTranPairQty += answRepQty * queryRepQty;
                }              
              }
              ia++;
              iq++;
            }
          }
        }
      }
      
      for (int iq=0; iq < queryEntry.mWordIds.length;++iq) {
        int queryWordId = queryEntry.mWordIds[iq];
        double collectProb = queryWordId >= 0 ? Math.max(outOfVocProb, fieldProbTable[queryWordId]) : outOfVocProb;
                                                  //answToQuestTran.getSourceWordProb(queryWordId)); 
        
        logScore += Math.log((1-lambda)*totTranProb[iq] +lambda*collectProb);        
      }
      
    }
  
    
    double logScoreQueryNorm = logScore / queryNorm;
    // For this feature, we can normalized by only query length
    double shareTranPairQtyQueryNorm = shareTranPairQty / queryNorm;
      

    DenseVector v = res.get(docId);
    
    if (v == null) {
      throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
    }    
    
    // Math.max avoid division by zero!
    double shareTranPairQtyNorm = shareTranPairQty / Math.max(1, queryEntry.mWordIdSeq.length * docEntry.mWordIdSeq.length);
  
    int fid = startFeatureId;
    if (useModel1) {
      v.set(fid++    , logScore);
      if (MODEL1_FIELD_FEATURE_QTY != 1) {
        throw new RuntimeException("Bug: wrong value of the constant MODEL1_FIELD_FEATURE_QTY");
      }
    }
    if (useModel1QueryNorm) {
      v.set(fid++    , logScoreQueryNorm);
      if (MODEL1_FIELD_FEATURE_QUERY_NORM_QTY != 1) {
        throw new RuntimeException("Bug: wrong value of the constant MODEL1_FIELD_FEATURE_QUERY_NORM_QTY");
      }      
    }
    if (useSimpleTran) {
      v.set(fid++, shareTranPairQty);
      v.set(fid++, shareTranPairQtyNorm);
      if (SIMPLE_TRAN_FIELD_FEATURE_QTY != 2) {
        throw new RuntimeException("Bug: wrong value of the constant SIMPLE_TRAN_FIELD_FEATURE_QTY");
      }     
    }
    if (useSimpleTranQueryNorm) {
      v.set(fid++, shareTranPairQtyQueryNorm);
      if (SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY != 1) {
        throw new RuntimeException("Bug: wrong value of the constant SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY");
      }      
    }

    if (PRINT_SCORES) {
      if (useModel1) System.out.println(String.format("Doc id %s %s: %g", docId, "IBM Model1 ", logScore));
      if (useModel1QueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "IBM Model1 (query-norm)", logScoreQueryNorm));
      if (useSimpleTran) System.out.println(String.format("Doc id %s %s: %g %g", docId, "simple tran ", shareTranPairQty, shareTranPairQtyNorm));
      if (useSimpleTranQueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "simple tran ", shareTranPairQtyQueryNorm));
    }
  }  
}
  

/**
 * Get all (IBM Model1 + simple) translation scores for one field.
 * 
 * <p>It is designed to worked only for the flipped translation table!</p>
 * 
 * @param fieldIndex        an in-memory field index
 * @param fieldId           a field identifier
 * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
 * @param arrDocIds         an array of document ids.
 * @param fieldName         a name of the field. 
 * @param minModel1Prob     a minimum Model1 probability for the field.
 * @param minSimpleTranProb a minimum simple tran. probability for the field
 * @param startFeatureId    an index/id of the first feature.
 * @param query             an actual query
 * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++) 
 * @param lambda            smoothing coefficient
 * @param outOfVocProb      a probability for the out-of-vocabulary word 
 * @param res               a result set to be updated.    
 * @throws Exception
 */
private void getFieldAllTranScoresFlipped(InMemForwardIndex fieldIndex,
                        int fieldId,
                        float[] fieldProbTable,
                        ArrayList<String> arrDocIds, 
                        String fieldName,
                        float minModel1Prob, float minSimpleTranProb, 
                        int startFeatureId,
                        String query,
                        GizaTranTableReaderAndRecoder answToQuestTran,
                        double lambda, 
                        double outOfVocProb, 
                        Map<String,DenseVector> res) throws Exception {    
  if (null == query) return;
  query = query.trim();
  if (query.isEmpty()) return;
   
  boolean useModel1         = useModel1Feature(fieldId);
  boolean useSimpleTran     = useSimpleTranFeature(fieldId);
  boolean useModel1QueryNorm     = useModel1FeatureQueryNorm(fieldId);
  boolean useSimpleTranQueryNorm = useSimpleTranFeatureQueryNorm(fieldId);
  
  DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
  
  
  int queryWordQty = queryEntry.mWordIds.length;
  
  float queryNorm = Math.max(1, queryWordQty);
    
  if (PRINT_SCORES)
    System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldAllTranScoresFlipped))");
  
  GizaOneWordTranRecs   queryTranRecs[] = new GizaOneWordTranRecs[queryWordQty];
  
  /*
   * We will read translation tables only one time per query.
   */
  for (int iq = 0; iq < queryWordQty; ++iq) {
    int queryWordId = queryEntry.mWordIds[iq];
    if (queryWordId < 0) continue; // out-of-vocab query words are ignored
    GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(queryWordId);
    /*
    if (null == tranRecs) continue;
    
    boolean found = false;
    for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
      if (tranRecs.mDstIds[k] == queryWordId) {
        found = true;

        if (PROB_SELF_TRAN - tranRecs.mProbs[k] > Float.MIN_NORMAL) {
          System.err.println("No self-tran probability for: id=" + queryWordId + "!");
          System.exit(1);
        }
        
        break;
      }
    }

    if (!found) {
      System.err.println("No self-tran probability for: id=" + queryWordId + "!");
      System.exit(1);
    }
    */
    
    queryTranRecs[iq] = tranRecs;    
  }
  
  for (String docId : arrDocIds) {
    DocEntry docEntry = fieldIndex.getDocEntry(docId);

    if (docEntry == null) {
      throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
    }

    double logScore = 0;
    float shareTranPairQty = 0;

    int answerQty = docEntry.mWordIds.length;    
    
    float [] aSourceWordProb = new float[docEntry.mWordIds.length];        
    float sum = 0;    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) 
      sum += docEntry.mQtys[ia];
    float invSum = 1/Math.max(1, sum);
      
    
    for (int ia=0; ia < docEntry.mWordIds.length; ++ia) {
      /*
      int answWordId = docEntry.mWordIds[ia];
      aSourceWordProb[ia] = fieldProbTable[answWordId];
      */
      aSourceWordProb[ia] = docEntry.mQtys[ia] * invSum;
    }

    for (int iqOuterLoop=0; iqOuterLoop < queryEntry.mWordIds.length;++iqOuterLoop) {
      float totTranProb = 0;

      int queryWordId = queryEntry.mWordIds[iqOuterLoop];
      int queryRepQty    = queryEntry.mQtys[iqOuterLoop];
      GizaOneWordTranRecs tranRecs = queryWordId >= 0 ? queryTranRecs[iqOuterLoop] : null;
            
      if (queryWordId >= 0 && tranRecs != null) {
        int[] tranWordIds = tranRecs.mDstIds;
        int tranRecsQty = tranRecs.mDstIds.length;
        int startIndex = 0;
        for (int ia = 0; ia < answerQty; ++ia) {
          int answWordId = docEntry.mWordIds[ia];
          int iq = Arrays.binarySearch(tranWordIds, startIndex, tranRecsQty, answWordId);
          if (iq >= 0) {
            float oneTranProb = tranRecs.mProbs[iq];
            int answRepQty = docEntry.mQtys[ia];

            if (oneTranProb >= minModel1Prob) {
              totTranProb += oneTranProb * aSourceWordProb[ia];
            }
            if (oneTranProb >= minSimpleTranProb) {
              shareTranPairQty += answRepQty * queryRepQty;
            }
          } else {
            // Because all the following answer word IDs are going to be even larger than
            // the current answWordId (and the array tranWordIds is also sorted),
            // we got the next entry point
            startIndex = -1 - iq;
            if (startIndex >= tranRecsQty)
              break;
          }
        }
      }
      
      double collectProb = queryWordId >=0 ? Math.max(outOfVocProb, fieldProbTable[queryWordId]) : outOfVocProb; 
      logScore += Math.log((1-lambda)*totTranProb +lambda*collectProb);
    }

    
    double logScoreQueryNorm = logScore / queryNorm;
    // For this feature, we can normalized by only query length
    double shareTranPairQtyQueryNorm = shareTranPairQty / queryNorm;
    
    DenseVector v = res.get(docId);
    
    if (v == null) {
      throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
    }    
    
    // Math.max avoid division by zero!
    double shareTranPairQtyNorm = shareTranPairQty / Math.max(1, queryEntry.mWordIdSeq.length * docEntry.mWordIdSeq.length);

    int fid = startFeatureId;
    if (useModel1) {
    	v.set(fid++    , logScore);
    	if (MODEL1_FIELD_FEATURE_QTY != 1) {
    		throw new RuntimeException("Bug: wrong value of the constant MODEL1_FIELD_FEATURE_QTY");
    	}
    }
    if (useModel1QueryNorm) {
      v.set(fid++    , logScoreQueryNorm);
      if (MODEL1_FIELD_FEATURE_QUERY_NORM_QTY != 1) {
        throw new RuntimeException("Bug: wrong value of the constant MODEL1_FIELD_FEATURE_QUERY_NORM_QTY");
      }      
    }
    if (useSimpleTran) {
	    v.set(fid++, shareTranPairQty);
	    v.set(fid++, shareTranPairQtyNorm);
      if (SIMPLE_TRAN_FIELD_FEATURE_QTY != 2) {
        throw new RuntimeException("Bug: wrong value of the constant SIMPLE_TRAN_FIELD_FEATURE_QTY");
      }	    
    }
    if (useSimpleTranQueryNorm) {
      v.set(fid++, shareTranPairQtyQueryNorm);
      if (SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY != 1) {
        throw new RuntimeException("Bug: wrong value of the constant SIMPLE_TRAN_FIELD_FEATURE_QUERY_NORM_QTY");
      }      
    }

    if (PRINT_SCORES) {
      if (useModel1) System.out.println(String.format("Doc id %s %s: %g", docId, "IBM Model1 ", logScore));
      if (useModel1QueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "IBM Model1 (query-norm)", logScoreQueryNorm));
      if (useSimpleTran) System.out.println(String.format("Doc id %s %s: %g %g", docId, "simple tran ", shareTranPairQty, shareTranPairQtyNorm));
      if (useSimpleTranQueryNorm) System.out.println(String.format("Doc id %s %s: %g", docId, "simple tran ", shareTranPairQtyQueryNorm));
    }
   }  
 }

  /**
   * Get TF-IDF scores for one field.
   * 
   * @param fieldIndex      an in-memory field index
   * @param similObj        an object that computes similarity.
   * @param arrDocIds       an array of document ids.
   * @param fieldName       a name of the field. 
   * @param featureId       an index/id of the feature.
   * @param query           an actual query
   * @param res             a result set to be updated.   * 
   * @throws Exception
   */
  private void getFieldScores(InMemForwardIndex fieldIndex,
                          QueryDocSimilarity    similObj,
                          ArrayList<String> arrDocIds, 
                          String fieldName,
                          int featureId,
                          String query,
                          Map<String,DenseVector> res) throws Exception {    
    if (null == query) return;
    query = query.trim();
    if (query.isEmpty()) return;
    
    DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
    
    
    if (PRINT_SCORES)
      System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldScores)");
   
    for (String docId : arrDocIds) {
      DocEntry docEntry = fieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
      
      float score = similObj.compute(queryEntry, docEntry);
      
      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      v.set(featureId, score);      
      if (PRINT_SCORES)
        System.out.println(String.format("Doc id %s %s: %g", docId, similObj.getName(), score)); 
    }      
  }

  /**
   * Get various embedding-based scores.
   * 
   * @param fieldIndex      an in-memory field index
   * @param fieldId         a field id
   * @param similObj        an object that computes similarity.
   * @param arrDocIds       an array of document ids.
   * @param fieldName       a name of the field. 
   * @param startFeatureId  an index/id of the first feature.
   * @param query           an actual query
   * @param res             a result set to be updated.   * 
   * @throws Exception
   */  
  
  private void getFieldEmbedScores(InMemForwardIndex fieldIndex,
                                 int fieldId,
                                 BM25SimilarityLucene similObj,
                                 ArrayList<String> arrDocIds, 
                                 String fieldName, 
                                 int startFeatureId, 
                                 String query,
                                 HashMap<String, DenseVector> res) throws Exception {

    if (null == query) return;
    query = query.trim();
    if (query.isEmpty()) return;
    
    DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));
    
    int embedQty = mWordEmbeds[fieldId].length;
    
    float [][] queryVecs            = new float[embedQty][];
    float [][] queryVecsIDFWeighted = new float[embedQty][];
    
    if (useAveragedEmbedFeatures(fieldId)) {
      for (int k = 0; k < embedQty; ++k) {
        if (useNonWghtAvgEmbed()) {
          queryVecs[k]            = mWordEmbeds[fieldId][k].getDocAverage(queryEntry, similObj, fieldIndex, 
                                            false, // don't multiply by IDF , 
                                            true // L2-normalize 
                                            );
        }
        queryVecsIDFWeighted[k] = mWordEmbeds[fieldId][k].getDocAverage(queryEntry, similObj, fieldIndex, 
                                                       true /* do multiply by IDF */, true /* L2-normalize */);
      }
    }
    
    if (PRINT_SCORES)
      System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldEmbedScores)");
   
    AbstractDistance distTypeL2 = AbstractDistance.create("l2"),
                     distTypeCosine = AbstractDistance.create("cosine");
    
    for (String docId : arrDocIds) {
      DocEntry docEntry = fieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }
            
      float[][] distMatrixL2 = null, distMatrixCosine = null;
          

      DenseVector v = res.get(docId);
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }      
      
      int featureId = startFeatureId;
      

      if (useWMDFeatures(fieldId)) {     
        if (null == distMatrixL2) 
          distMatrixL2 = DistanceFunctions.compDistMatrix(distTypeL2, queryEntry, docEntry, mWordEmbeds[fieldId][0]);
        
        float scores[] = DistanceFunctions.compEMDLike(queryEntry, docEntry, distMatrixL2);
      

        StringBuffer sbScores = new StringBuffer();
        for (int k = 0; k < scores.length; ++k) {
          v.set(featureId++, scores[k]);
          sbScores.append(scores[k]+ " ");
        }
      
        if (PRINT_SCORES)
          System.out.println(String.format("WMD: Doc id %s %s: %s", docId, distTypeL2.getName(), 
                                          sbScores));
      }
      if (useLCSEmbedFeatures(fieldId)) {
        if (null == distMatrixCosine)
          distMatrixCosine = DistanceFunctions.compDistMatrix(distTypeCosine, queryEntry, docEntry, mWordEmbeds[fieldId][0]);
        
        // LCSLike *MUST* be used only with cosines
        float scores[] = DistanceFunctions.compLCSLike(distMatrixCosine, LCS_WORD_EMBED_THRESH);
        float scoresNorm[] = new float[scores.length];
        
        if (queryEntry.mWordIds.length > 0)
        for (int k = 0; k < scores.length; ++k) {
          scoresNorm[k] = scores[k] / queryEntry.mWordIds.length;
        }
        // If query length is zero, all scoresNorm are zero by default
        
        StringBuffer sbScores = new StringBuffer();
        for (int k = 0; k < scores.length; ++k) {
          v.set(featureId++, scores[k]);
          sbScores.append(scores[k]+ " ");
          v.set(featureId++, scoresNorm[k]);
          sbScores.append(scoresNorm[k]+ " ");
        }
        
        if (PRINT_SCORES)
          System.out.println(String.format("LCS: Doc id %s %s: %s", docId, distTypeL2.getName(), 
                                          sbScores));
                
      }
      if (useAveragedEmbedFeatures(fieldId)) {
        for (int k = 0; k < embedQty; ++k) {
          
          float [] docVec = null;
          if (useNonWghtAvgEmbed()) {
              docVec = mWordEmbeds[fieldId][k].getDocAverage(docEntry, similObj, fieldIndex, 
                                       false, // don't multiply by IDF  
                                       true // L2-normalize 
                                       );
          }
          float [] docVecIDFWeighted = 
              mWordEmbeds[fieldId][k].getDocAverage(docEntry, similObj, fieldIndex,
                                       true /* do multiply by IDF */, true /* L2-normalize */);
                                       
          
          /*
           * The more similar are the vectors, the larger are the values of v1/v2!
           */
          float v1 = 0;
          if (useNonWghtAvgEmbed()) {
            v1 = 2 - distTypeCosine.compute(queryVecs[k], docVec);
          }
          float   v2 = 2 - distTypeCosine.compute(queryVecsIDFWeighted[k], docVecIDFWeighted);
          
          if (useNonWghtAvgEmbed()) {
            v.set(featureId++, v1);
          }
          v.set(featureId++, v2);
  
          if (PRINT_SCORES)
            if (useNonWghtAvgEmbed()) {
              System.out.println(String.format("AVG: Doc id %s %s=%g %s=%g embed id=%d", 
                                                docId, 
                                                distTypeL2.getName(), v1, 
                                                distTypeCosine.getName(), v2,
                                                k));
            } else {
              System.out.println(String.format("AVG: Doc id %s %s=%g embed id=%d", 
                  docId, 
                  distTypeCosine.getName(), v2,
                  k));
            }

        }        
      }      
      if (useAveragedEmbedBM25Features(fieldId)) {
        if (null == distMatrixCosine)
          distMatrixCosine = DistanceFunctions.compDistMatrix(distTypeCosine, queryEntry, docEntry, mWordEmbeds[fieldId][0]);
        
        float scores[] = similObj.computeEmbed(distMatrixCosine, queryEntry, docEntry);

        v.set(featureId++, scores[0]);
        v.set(featureId++, scores[1]);
        if (AVERAGED_EMBEDBM25_FEATURE_QTY != 2 ) {
          throw new RuntimeException("Wrong value of constant AVERAGED_EMBEDBM25_FEATURE_QTY");
        }
        if (PRINT_SCORES)
          System.out.println(String.format("Embed BM25: Doc id %f %f", scores[0], scores[1]));

      }
    }

  }
  
   
  /**
   * Compute not so simple translation scores for one field based only on composed JS-Divergence.
   * 
   * <p>Note that flipping of the translation table may affect the outcome,
   * however, not in our experimental setup, when we obtain symmetrized translation
   * tables. 
   * </p>
   * 
   * @param fieldIndex        an in-memory field index
   * @param fieldId           a field identifier
   * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
   * @param arrDocIds         an array of document ids.
   * @param fieldName         a name of the field. 
   * @param minProb           a minimum translation probability to be taken into account.
   * @param startFeatureId    an index/id of the first feature.
   * @param query             an actual query
   * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++) 
   * @param res               a result set to be updated.    
   * @throws Exception
   */
  private void getFieldJSDCompositeScores(InMemForwardIndex fieldIndex,
                          int fieldId,
                          float[] fieldProbTable,
                          ArrayList<String> arrDocIds, 
                          String fieldName,
                          float minProb, 
                          int startFeatureId,
                          String query,
                          GizaTranTableReaderAndRecoder answToQuestTran,
                          Map<String,DenseVector> res) throws Exception {    
    if (null == query) return;
    query = query.trim();
    if (query.isEmpty()) return;
    
    DocEntry queryEntry = fieldIndex.createDocEntry(query.split("\\s+"));

    SparseVector[] queryEmbedVectorsL1Norm = new SparseVector[mHighOrderModels.size()];

    ArrayList<HashIntObjMap<SparseVector>> highOrderFieldModels = mHighOrderModels.get(fieldId);
    
    if (highOrderFieldModels == null)
      throw new Exception("Bug: not high-order models for field=" + mFieldNames[fieldId] + " fieldId=" + fieldId);
    
    for (int k = 0; k < highOrderFieldModels.size(); ++k) {
      queryEmbedVectorsL1Norm[k] = SparseEmbeddingReaderAndRecorder.
          createCompositeWordEmbed(fieldIndex, highOrderFieldModels.get(k), queryEntry);
    }
    
    
      
    if (PRINT_SCORES)
      System.out.println("InMemIndex Field: '" + fieldName + "' (getFieldNotSoSimpleTranScoresJSDCompOnly))");
   
    for (String docId : arrDocIds) {
      DocEntry docEntry = fieldIndex.getDocEntry(docId);
      
      if (docEntry == null) {
        throw new Exception("Inconsistent data or bug: can't find document with id ='" + docId + "'");
      }      

      DenseVector v = res.get(docId);
      
      if (v == null) {
        throw new Exception(String.format("Bug, cannot retrieve a vector for docId '%s' from the result set", docId));
      }
      
      int fid = startFeatureId;      

      for (int k = 0; k < highOrderFieldModels.size(); ++k) {
        SparseVector docEmbedL1Norm = SparseEmbeddingReaderAndRecorder.
                                          createCompositeWordEmbed(fieldIndex, highOrderFieldModels.get(k), docEntry);             
        
        // Let's take the negative value, then the larger score2, the closer are documents
        double score2 = -Math.sqrt(DistanceFunctions.computeJSDiv(queryEmbedVectorsL1Norm[k], docEmbedL1Norm));
  
        v.set(fid++    , score2);
    
        if (PRINT_SCORES) {
          System.out.println(String.format("Doc id %s %s (k=%d): %g", docId, "scoreJSDivSqrt   ", k, score2));        
        }
        
      }
      
      
      if (fid - startFeatureId != mJSDCompositeFeatQty) {
        throw new RuntimeException("Bug: apparently wrong value of the constant JSD_COMPOSITE_FEATURE_QTY");
      }      

    }  
  }      
  
  /**
   * Creates a simple frequency vector (not normalized) from a document entry.
   * 
   * @param   fieldIndex    an in-memory field index.
   * @param   e             a document entry.
   * 
   * @return a simple frequency vector (not normalized) from a document entry.
   */
  public SparseVector createFreqVector(InMemForwardIndex fieldIndex, DocEntry e) {
    int qty =0;
    for (int i = 0; i < e.mQtys.length; ++i) {
      int wordId = e.mWordIds[i];
      if (wordId >= 0) qty++; 
    }
    int    ids[] = new int[qty];
    double vals[] = new double[qty];
    int indx = -1;
    for (int i = 0; i < e.mQtys.length; ++i) {
      int wordId = e.mWordIds[i];
      if (wordId >= 0) {
        ++indx;
        ids[indx] = wordId;
        vals[indx]=e.mQtys[i];
      }
    }
    return new SparseVector(fieldIndex.getMaxWordId()+1, ids, vals, false);
  }
  
  /**
   * Create a single L1-normalized word embedding based on translation probabilities, results are cached.
   * The idea is taken from the following paper (though the implementation is a bit different):
   * Higher-order Lexical Semantic Models for Non-factoid Answer Reranking.
   * Fried, et al. 2015.
   * 
   * @param fieldIndex        an in-memory field index
   * @param minProb           a minimum translation probability
   * @param fieldProbTable    field-specific word probabilities for the answer vocabulary.
   * @param answToQuestTran   answer-to-question translation probabilities (computed by GIZA or GIZA++)
   * @param wordId            a word ID
   * @return
   */
  public synchronized SparseVector createTranBasedWordEmbedding(InMemForwardIndex             fieldIndex,
                                                                float                         minProb, 
                                                                float[]                       fieldProbTable,
                                                                GizaTranTableReaderAndRecoder answToQuestTran,                                               
                                                                int                           wordId) {
    if (wordId < 0) return null;
    SparseVector res = mTranWordEmbedCache.get(wordId);
    if (res != null) return res;
    
    GizaOneWordTranRecs tranRecs = answToQuestTran.getTranProbs(wordId);
    
    if (tranRecs != null) {
      int qty = 0;
      double norm = 0;
      for (float p : tranRecs.mProbs ) { 
        if (p >= minProb) {
          ++qty;
          norm += p;
        }
      }
      norm = 1.0 / norm;
      
      int     ids[]  = new int[qty];
      double  vals[] = new double[qty];
      int     indx   = 0;
      
      for (int i = 0; i < tranRecs.mDstIds.length; ++i) {
        double p = tranRecs.mProbs[i];
        if (p >= minProb) {
          ids[indx]  = tranRecs.mDstIds[i];
          vals[indx] = p * norm;
          indx++;
        }
      }
      res = new SparseVector(fieldIndex.getMaxWordId()+1, ids, vals, false);
    }
    
    mTranWordEmbedCache.put(wordId, res);
    return res;
  }
   
  
  InMemForwardIndex mFieldIndex[] = new InMemForwardIndex[FeatureExtractor.mFieldNames.length];

  /* START OF FIELD PARAMS */
  
  protected static boolean[] mFlippedTranTableFieldUse = {
    true,  // text
    true,  // text_unlemm
    false, // bigram
    false, // srl
    false, // srl_lab
    false, // dep
    false, // wnss
    false, // qfeat_all
    true,  // text_alias1
   };
  
  
  
  protected static float[] mModel1LambdaDefault = {
                                0.1f, // text 
                                0.1f, // text_unlemm
                                0.4f,    // bigram 
                                0.075f,  // srl
                                0.5f,    // srl_lab 
                                0.075f,  // dep 
                                0.1f,    // wnss
                                0.1f,    // qfeat_all
                                0.1f,  // text_alias1
                              };
  
  protected static float[] mProbSelfTranDefault = {
                                0.05f, // text
                                0.01f, // text_unlemm
                                DEFAULT_PROB_SELF_TRAN, // bigram
                                DEFAULT_PROB_SELF_TRAN, // srl
                                DEFAULT_PROB_SELF_TRAN, // srl_lab
                                DEFAULT_PROB_SELF_TRAN, // dep
                                DEFAULT_PROB_SELF_TRAN, // wnss
                                0.001f, // qfeat_all
                                0.05f, // text_alias1
   };

  // For Model1 translation features, ignore translation probabilities smaller than this value
  protected static float[] mMinModel1ProbDefault = {
                               2.5e-3f, // text 
                               2.5e-3f, // text_unlemm
                               1e-5f, // bigram 
                               1e-5f, // srl
                               1e-5f, // srl_lab 
                               1e-5f, // dep 
                               1e-4f, // wnss 
                               1e-3f, // qfeat_all
                               2.5e-3f, // text_alias1 
                             };

  protected static float[] mMinJSDCompositeProbDefault =   {
                              1e-2f, // text
                              1e-2f, // text_unlemm
                              1e-2f, // bigram 
                              1e-2f, // srl
                              1e-2f, // srl_lab 
                              1e-2f, // dep 
                              1e-2f, // wnss
                              1e-2f, // qfeat_all
                              1e-2f, // text_alias1
  };
  
 // For smiple tran feature, ignore translation probabilities smaller than this value
  protected static float[] mMinSimpleTranProbDefault = {
                                2.5e-3f, // text
                                2.5e-3f, // text_unlemm
                                1e-5f,   // bigram 
                                1e-5f,   // srl 
                                1e-5f,   // srl_lab 
                                1e-5f,   // dep 
                                1e-2f,   // wnss
                                1e-3f,   // qfeat_all
                                2.5e-3f, // text_alias1
                               };
    
  /* END OF FIELD PARAMS */
  
  protected float maFieldProbTable[][] = new float[mFieldNames.length][];

  final BM25SimilarityLucene        mBM25Similarity[] = new BM25SimilarityLucene[FeatureExtractor.mFieldNames.length];
  final DefaultSimilarityLucene     mDefaultSimilarity[] = new DefaultSimilarityLucene[FeatureExtractor.mFieldNames.length];
  final BM25SimilarityLuceneNorm    mBM25SimilarityNorm[] = new BM25SimilarityLuceneNorm[FeatureExtractor.mFieldNames.length];
  final DefaultSimilarityLuceneNorm mDefaultSimilarityNorm[] = new DefaultSimilarityLuceneNorm[FeatureExtractor.mFieldNames.length];
  final CosineTextSimilarity        mCosineTextSimilarity[] = new CosineTextSimilarity[FeatureExtractor.mFieldNames.length];
  
  GizaTranTableReaderAndRecoder   [] maAnswToQuestTran = new GizaTranTableReaderAndRecoder[mFieldNames.length];

  protected ArrayList<ArrayList<HashIntObjMap<SparseVector>>> mHighOrderModels = null;

  private final String    mDirTranPrefix;
  private final int       mGizaIterQty;
  private final String    mIndexDir;
  private final String    mEmbedDir;
  private final String[]  mEmbedFiles;
  private final String[]  mHighOrderModelFiles;  

  private int mFieldScoreFeatQty  = 0;
  private int mModel1FeatQty = 0;
  private int mSimpleTranFeatQty = 0;
  private int mJSDCompositeFeatQty = 0;
  private int mLCSFeatQty = 0;
  private int mOverallMatchFeatQty = 0;
     
  
  protected int                           mAveragedEmbedFeatureQty = 0;
  protected int                           mAveragedEmbedBM25FeatureQty = 0;
  protected int                           mLCSEmbedFeatQty = 0;
  protected int                           mWMDFeatQty = 0;
    
  protected final EmbeddingReaderAndRecoder[][] mWordEmbeds = new EmbeddingReaderAndRecoder[FeatureExtractor.mFieldNames.length][];

  private HashIntObjMap<SparseVector>  mTranWordEmbedCache =       
                                                          HashIntObjMaps.<SparseVector>newMutableMap(INIT_VOCABULARY_SIZE);
}
