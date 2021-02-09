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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.concurrent.*;

import javax.annotation.Nullable;

import no.uib.cipr.matrix.DenseVector;

import org.apache.commons.cli.*;
import org.apache.commons.math3.stat.descriptive.SynchronizedSummaryStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.*;
import edu.cmu.lti.oaqa.flexneuart.letor.CompositeFeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.letor.DataPointWrapper;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import ciir.umass.edu.learning.*;


class BaseProcessingUnit {
  public static Object              mWriteLock = new Object();
  protected final BaseQueryApp      mAppRef;
  
  BaseProcessingUnit(BaseQueryApp appRef) {
    mAppRef    = appRef;
  }  
  
  public void procQuery(CandidateProvider candProvider, int queryNum) throws Exception {
    Map<String, String>    queryFields = mAppRef.mParsedQueries.get(queryNum);

    String queryId = queryFields.get(Const.TAG_DOCNO);
            
    // 2. Obtain results
    long start = System.currentTimeMillis();
    
    CandidateInfo qres = null;
    
    if (mAppRef.mResultCache != null) 
      qres = mAppRef.mResultCache.getCacheEntry(queryId);
    if (qres == null) {            

      String text = queryFields.get(Const.TEXT_FIELD_NAME);
      if (text != null) {
        // This was a workaround for a pesky problem: didn't previously notice that the string
        // n't (obtained by tokenization of can't is indexed. Querying using this word
        // add a non-negligible overhead (although this doesn't affect overall accuracy)
        // However, we believe data processing scripts should now always remove these extra stop-words
        //queryFields.put(Const.TEXT_FIELD_NAME, CandidateProvider.removeAddStopwords(text));
        queryFields.put(Const.TEXT_FIELD_NAME, text);
      }
      qres = candProvider.getCandidates(queryNum, queryFields, mAppRef.mMaxCandRet);
      if (mAppRef.mResultCache != null) 
        mAppRef.mResultCache.addOrReplaceCacheEntry(queryId, qres);
    }
    CandidateEntry [] cands = qres.mEntries;
    // Let's re-rank candidates just in case the candidate provider fails to retrieve entries in the right order
    Arrays.sort(cands);
    
    long end = System.currentTimeMillis();
    long searchTimeMS = end - start;
    
    mAppRef.logger.info(
        String.format("Obtained results for the query # %d queryId='%s', the search took %d ms, we asked for max %d entries got %d", 
                      queryNum, queryId, searchTimeMS, mAppRef.mMaxCandRet, cands.length));
    
    mAppRef.mQueryTimeStat.addValue(searchTimeMS);
    mAppRef.mNumRetStat.addValue(qres.mNumFound);
    
    // allDocFeats will be first created by an intermediate re-ranker (if it exists).
    // If there is a final re-ranker, it will overwrite previously created features.
    Map<String, DenseVector> allDocFeats = null;
    Integer maxNumRet = mAppRef.mMaxNumRet;
            
    // 3. If necessary carry out an intermediate re-ranking
    if (mAppRef.mExtrInterm != null) {
      // Compute features once for all documents using an intermediate re-ranker
      start = System.currentTimeMillis();
      allDocFeats = mAppRef.mExtrInterm.getFeatures(cands, queryFields);
      
      DenseVector intermModelWeights = mAppRef.mModelInterm;

      for (int rank = 0; rank < cands.length; ++rank) {
        CandidateEntry e = cands[rank];
        DenseVector feat = allDocFeats.get(e.mDocId);
        if (feat.size() != intermModelWeights.size()) {
          throw new Exception(String.format("Feature and linear model dim. mismatch in the intermedaite re-ranker, features: %d model: %d",
                                            feat.size(), intermModelWeights.size()));
        }
        e.mScore = (float) feat.dot(intermModelWeights);
        if (Float.isNaN(e.mScore)) {
          if (Float.isNaN(e.mScore)) {
            mAppRef.logger.info("DocId=" + e.mDocId + " queryId=" + queryId);
            mAppRef.logger.info("NAN scores, feature vector:");
            mAppRef.logger.info(feat.toString());
            mAppRef.logger.info("NAN scores, feature weights:");
            mAppRef.logger.info(intermModelWeights.toString());
            throw new Exception("NAN score encountered (intermediate reranker)!");
          }
        }
      }
      // Re-sorting after updating scores
      Arrays.sort(cands);
      // We may now need to update allDocIds and resultsAll to include only top-maxNumRet entries!
      if (cands.length > maxNumRet) {
        CandidateEntry resultsAllTrunc[] = Arrays.copyOf(cands, maxNumRet);
        cands = resultsAllTrunc;          
      }
      end = System.currentTimeMillis();
      long rerankIntermTimeMS = end - start;
      mAppRef.logger.info(
          String.format("Intermediate-feature generation & re-ranking for the query # %d queryId='%s' took %d ms", 
                         queryNum, queryId, rerankIntermTimeMS));
      mAppRef.mIntermRerankTimeStat.addValue(rerankIntermTimeMS);          
    }
            
    // 4. If QRELs are specified, we need to save results only for subsets that return a relevant entry. 
    //    Let's see what's the rank of the highest ranked entry. 
    //    If, e.g., the rank is 10, then we discard subsets having less than top-10 entries.
    int minRelevRank = Integer.MAX_VALUE;
    if (mAppRef.mQrels != null) {
      for (int rank = 0; rank < cands.length; ++rank) {
        CandidateEntry e = cands[rank];
        String label = mAppRef.mQrels.get(queryId, e.mDocId);
        e.mRelevGrade = CandidateProvider.parseRelevLabel(label);
        if (e.mRelevGrade >= 1 && minRelevRank == Integer.MAX_VALUE) {
          minRelevRank = rank;
        }
      }
    } else {
      minRelevRank = 0;
    }
    
    Ranker modelFinal = mAppRef.mModelFinal;
    int rerankQty = Math.min(cands.length, mAppRef.mMaxFinalRerankQty);
    // 5. If the final re-ranking model is specified, let's re-rank again and save all the results
    if (mAppRef.mExtrFinal!= null && rerankQty > 0 && cands.length > 0) {
      
      if (cands.length > maxNumRet) {
        throw new RuntimeException("Bug or you are using old/different cache: cands.size()=" + cands.length + " > maxNumRet=" + maxNumRet);
      }
      // Compute features once for all documents using a final re-ranker.
      // Note, however, we might choose to re-rank only top candidates not all of them
      
      start = System.currentTimeMillis();
      

      CandidateEntry candsToRerank[] = Arrays.copyOf(cands, rerankQty);
      allDocFeats = mAppRef.mExtrFinal.getFeatures(candsToRerank, queryFields);
      
      if (modelFinal != null) {
        DataPointWrapper featRankLib = new DataPointWrapper();
        float minTopRerankScore = Float.MAX_VALUE;
        // Because candidates are sorted, this is going to be the
        // smallest score among rerankQty top candidates.
        float minTopOrigScore = cands[rerankQty - 1].mScore;
        for (int rank = 0; rank < rerankQty; ++rank) {
          CandidateEntry e = cands[rank];
          DenseVector feat = allDocFeats.get(e.mDocId);
          // It looks like eval is thread safe in RankLib
          featRankLib.assign(feat);
          e.mScore = (float) modelFinal.eval(featRankLib);
          // Re-ranked scores aren't guaranteed to be ordered, 
          // so the last candidate won't necessary have the lowest score.
          // We need to compute the minimum explicitly.
          minTopRerankScore = Math.min(e.mScore, minTopRerankScore);
          if (Float.isNaN(e.mScore)) {
            if (Float.isNaN(e.mScore)) {
              mAppRef.logger.info("DocId=" + e.mDocId + " queryId=" + queryId);
              mAppRef.logger.info("NAN scores, feature vector:");
              mAppRef.logger.info(feat.toString());
              throw new Exception("NAN score encountered (intermediate reranker)!");
            }
          }
        }
        if (rerankQty < cands.length) {

          // If the we don't re-rank tail entries, their scores still have to be adjusted
          // so that the order is preserved and all the scores are below the minimum score
          // for all the re-ranked entries
          for (int rank = rerankQty; rank < cands.length; ++rank) {
            CandidateEntry e = cands[rank];
            float origScore = e.mScore;
            // currScore must be <= minTopOrigScore, so we could get only
            // smaller values compared to the top-k cohort
            // Note the brackets: without them the lack of associativity 
            // due to floating point errors may lead to the sanity check failure
            e.mScore = minTopRerankScore + (origScore - minTopOrigScore);
            if (e.mScore > minTopRerankScore + 1e-6) {
              mAppRef.logger
                  .info(String.format("orig score: %f updated score: %f minTopRerankScore: %f minTopOrigScore: %f",
                      origScore, e.mScore, minTopRerankScore, minTopOrigScore));
              throw new RuntimeException("Shouldn't happen: it's a ranking bug!");
            }
          }
        } 
      }
      
      // Re-sorting after updating scores
      Arrays.sort(cands);
      
      end = System.currentTimeMillis();
      long rerankFinalTimeMS = end - start;
      mAppRef.logger.info(
          String.format("Final-feature generation & re-ranking for the query # %d queryId='%s', final. reranking took %d ms", 
                        queryNum, queryId, rerankFinalTimeMS));
      mAppRef.mFinalRerankTimeStat.addValue(rerankFinalTimeMS);                        
    }
    
    
    // Now that all documents are re-ranked we simply output them
    

    for (int k = 0; k < mAppRef.mNumRetArr.size(); ++k) {
      int numRet = mAppRef.mNumRetArr.get(k);
      if (numRet >= minRelevRank) {
        CandidateEntry resultsCurr[] = Arrays.copyOf(cands, Math.min(numRet, cands.length));
        // We previously used to-resort results here to be compliant with some previous publications.
        // It is clearly a counter productive thing to do.
        //Arrays.sort(resultsCurr);
        synchronized (mWriteLock) {
          mAppRef.procResults(
              mAppRef.mRunId,
              queryId,
              queryFields,
              resultsCurr,
              numRet,
              allDocFeats
           );
        }
      }
    }                
  }
}

class BaseQueryAppProcessingWorker implements Runnable  {
  private final BaseProcessingUnit mProcUnit;
  private final int                mQueryNum;
  
  BaseQueryAppProcessingWorker(BaseQueryApp appRef,
                               int          queryId) {
    mProcUnit = new BaseProcessingUnit(appRef);
    mQueryNum   = queryId;    
  }
  
  private static int                              mUsedProvQty = 0;
  private static HashMap<Long,CandidateProvider>  mProvMapping = new HashMap<Long,CandidateProvider>(); 
  
  /*
   * This function retrieves a candidate provider for a given thread. It assumes that the number of threads
   * is equal to the number of provider entries.
   */
  private static synchronized CandidateProvider getCandProvider(CandidateProvider[] candProvList)  {
    long threadId = Thread.currentThread().getId();
    CandidateProvider cand = mProvMapping.get(threadId);
    if (cand != null) return cand;
    if (mUsedProvQty >= candProvList.length)
      throw new RuntimeException("Probably a bug: I am out of candidate providers, did you create more threads than provider entries?");
    cand = candProvList[mUsedProvQty];
    mUsedProvQty++;
    mProvMapping.put(threadId, cand);
    return cand;
  }
  
  @Override
  public void run() {
    
    try {
      CandidateProvider candProvider = getCandProvider(mProcUnit.mAppRef.mCandProviders);
      
      mProcUnit.procQuery(candProvider, mQueryNum);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Unhandled exception: " + e + " ... exiting");
      System.exit(1);
    }
  }
}

class BaseQueryAppProcessingThread extends Thread  {
  private final BaseProcessingUnit mProcUnit;
  private final int                mThreadId;
  private final int                mThreadQty;
  
  BaseQueryAppProcessingThread(BaseQueryApp appRef,
                               int          threadId,
                               int          threadQty) {
    mProcUnit = new BaseProcessingUnit(appRef);
    mThreadId  = threadId;
    mThreadQty = threadQty; 
  }

  @Override
  public void run() {
    
    try {
      mProcUnit.mAppRef.logger.info("Thread id=" + mThreadId + " is created, the total # of threads " + mThreadQty);
      
      CandidateProvider candProvider = mProcUnit.mAppRef.mCandProviders[mThreadId];

      for (int iq = 0; iq < mProcUnit.mAppRef.mParsedQueries.size(); ++iq)
        if (iq % mThreadQty == mThreadId) 
          mProcUnit.procQuery(candProvider, iq);
      
      mProcUnit.mAppRef.logger.info("Thread id=" + mThreadId + " finished!");
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Unhandled exception: " + e + " ... exiting");
      System.exit(1);
    }    
  }
}

/**
 * This class provides a basic functionality related to
 * reading parameters, initializing shared resources, creating
 * result providers, etc; It has abstract function(s) that are
 * supposed to implemented in child classes.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class BaseQueryApp {

  /**
   * A child class implements this function where it calls {@link #addCandGenOpts(boolean, boolean, boolean)},
   * {@link #addResourceOpts(boolean)}, and {@link #addLetorOpts(boolean, boolean)} with appropriate parameters,
   * as well as adds custom options.
   * 
   */
  abstract void addOptions();
  
  /**
   * A child class implements this function to read values of custom options.
   */
  abstract void procCustomOptions();
  
  /**
   * A child class initializes, e.g., output file(s) in this function.
   * @throws Exception 
   */
  abstract void init() throws Exception;
  /**
   * A child class finializes execution, e.g., closes output files(s) in this function.
   * @throws Exception 
   */
  abstract void fin() throws Exception;
  /**
   * A child class will process results in this function, e.g, it will save them to the output file;
   * The procRecults() function needn't be synchronized</b>, class {@link BaseQueryAppProcessingThread} 
   * will take care to sync. 
   * @param docFields 
   * 
   * @param   runId
   *              an ID of the run
   * @param   queryId
   *              a query ID
   * @param   docFields
   *              a map, where keys are field names, while values represent
   *              values of indexable fields.
   * @param   scoredDocs
   *              a list of scored document entries
   * @param   numRet
   *              The result set will be generated for this number of records.
   * @param   docFeats
   *              a list of document entry features (may be NULL)
   *      
   */
  abstract void procResults(
      String                              runId,
      String                              queryId,
      Map<String, String>                 docFields, 
      CandidateEntry[]                    scoredDocs,
      int                                 numRet,
      @Nullable Map<String, DenseVector>  docFeats
      ) throws Exception;
    
  
  final Logger logger = LoggerFactory.getLogger(BaseQueryApp.class);
  
  void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(mAppName, mOptions);      
    System.exit(1);
  }
  void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }  

  
  /**
   * Adds options related to candidate generation.  
   * 
   * @param onlyLucene
   *                    if true, we don't allow to specify the provider type and only use Lucene
   * @param multNumRetr
   *                    if true, an app generates results for top-K sets of various sizes
   * @param useQRELs
   *                    if true, an app uses QRELs
   * @param useThreadQty
   *                    if true, an app allows to specify the number of threads.                     
   */
  void addCandGenOpts(boolean onlyLucene,
                      boolean multNumRetr,
                      boolean useQRELs,
                      boolean useThreadQty) {
    mMultNumRetr = multNumRetr;
    mOnlyLucene = onlyLucene;

    mOptions.addOption(CommonParams.CAND_PROVID_ADD_CONF_PARAM,   null, true,  CommonParams.CAND_PROVID_ADD_CONF_DESC);
    if (onlyLucene) {
      mOptions.addOption(CommonParams.PROVIDER_URI_PARAM,      null, true, CommonParams.LUCENE_INDEX_LOCATION_DESC);
    } else {
      mOptions.addOption(CommonParams.CAND_PROVID_PARAM,       null, true, CandidateProvider.CAND_PROVID_DESC);
      mOptions.addOption(CommonParams.PROVIDER_URI_PARAM,      null, true, CommonParams.PROVIDER_URI_DESC);
    }
    
    mOptions.addOption(CommonParams.RUN_ID_PARAM,              null, true, CommonParams.RUN_ID_DESC);
    
    mOptions.addOption(CommonParams.QUERY_CACHE_FILE_PARAM,    null, true, CommonParams.QUERY_CACHE_FILE_DESC);
    mOptions.addOption(CommonParams.QUERY_FILE_PARAM,          null, true, CommonParams.QUERY_FILE_DESC);
    mOptions.addOption(CommonParams.MAX_NUM_RESULTS_PARAM,     null, true, mMultNumRetr ? 
                                                                             CommonParams.MAX_NUM_RESULTS_DESC : 
                                                                             "A number of candidate records (per-query)");
    mOptions.addOption(CommonParams.MAX_NUM_QUERY_PARAM,       null, true, CommonParams.MAX_NUM_QUERY_DESC);

    mUseQRELs = useQRELs;
    if (mUseQRELs)
      mOptions.addOption(CommonParams.QREL_FILE_PARAM,         null, true, CommonParams.QREL_FILE_DESC);

    if (useThreadQty)
      mOptions.addOption(CommonParams.THREAD_QTY_PARAM,        null, true, CommonParams.THREAD_QTY_DESC);
    
    mOptions.addOption(CommonParams.SAVE_STAT_FILE_PARAM,      null, true, CommonParams.SAVE_STAT_FILE_DESC);
    mOptions.addOption(CommonParams.USE_THREAD_POOL_PARAM,     null, false, CommonParams.USE_THREAD_POOL_DESC);
  }
  
  /**
   * Adds options related to resource initialization.
   * 
   */
  void addResourceOpts() {    
    mOptions.addOption(CommonParams.FWDINDEX_PARAM,            null, true,  CommonParams.FWDINDEX_DESC);    
    mOptions.addOption(CommonParams.GIZA_ROOT_DIR_PARAM,       null, true,  CommonParams.GIZA_ROOT_DIR_DESC);
    mOptions.addOption(CommonParams.EMBED_DIR_PARAM,           null, true,  CommonParams.EMBED_DIR_DESC);         
  }
  
  /**
   * Adds options related to LETOR (learning-to-rank).
   */
  void addLetorOpts(boolean useIntermModel, boolean useFinalModel) {
    
    mOptions.addOption(CommonParams.EXTRACTOR_TYPE_FINAL_PARAM,     null, true,  CommonParams.EXTRACTOR_TYPE_FINAL_DESC);
    mOptions.addOption(CommonParams.EXTRACTOR_TYPE_INTERM_PARAM,    null, true,  CommonParams.EXTRACTOR_TYPE_INTERM_DESC);
    
    mUseIntermModel = useIntermModel;
    mUseFinalModel = useFinalModel;
    
    if (mUseIntermModel) {
      mOptions.addOption(CommonParams.MODEL_FILE_INTERM_PARAM, null, true, CommonParams.MODEL_FILE_INTERM_DESC);
      mOptions.addOption(CommonParams.MAX_CAND_QTY_PARAM,      null, true, CommonParams.MAX_CAND_QTY_DESC);
    }
    if (mUseFinalModel) {
      mOptions.addOption(CommonParams.MODEL_FILE_FINAL_PARAM,  null, true, CommonParams.MODEL_FILE_FINAL_DESC);
    }
    mOptions.addOption(CommonParams.MAX_FINAL_RERANK_QTY_PARAM, null, true, CommonParams.MAX_CAND_QTY_DESC);
  }
  
  /**
   * Parses arguments and reads specified options (additional options can be read by subclasses)
   * @throws Exception 
   */
  void parseAndReadOpts(String args[]) throws Exception {
    mCmd = mParser.parse(mOptions, args);

    if (mOnlyLucene)
      mCandProviderType = CandidateProvider.CAND_TYPE_LUCENE;
    else { 
      mCandProviderType = mCmd.getOptionValue(CommonParams.CAND_PROVID_PARAM);
      if (null == mCandProviderType) showUsageSpecify(CommonParams.CAND_PROVID_DESC);
    }
    mCandProviderConfigName = mCmd.getOptionValue(CommonParams.CAND_PROVID_ADD_CONF_PARAM);
    mRunId = mCmd.getOptionValue(CommonParams.RUN_ID_PARAM);
    if (mRunId == null) showUsageSpecify(CommonParams.RUN_ID_PARAM);
    
    mProviderURI = mCmd.getOptionValue(CommonParams.PROVIDER_URI_PARAM);
    if (null == mProviderURI) showUsageSpecify(CommonParams.PROVIDER_URI_DESC);              
    mQueryFile = mCmd.getOptionValue(CommonParams.QUERY_FILE_PARAM);
    if (null == mQueryFile) showUsageSpecify(CommonParams.QUERY_FILE_DESC);
    {
      String tmpn = mCmd.getOptionValue(CommonParams.MAX_NUM_QUERY_PARAM);
      if (tmpn != null) {
        try {
          mMaxNumQuery = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Maximum number of queries isn't integer: '" + tmpn + "'");
        }
      }
    }
    {
      String tmpn = mCmd.getOptionValue(CommonParams.MAX_NUM_RESULTS_PARAM);
      if (null == tmpn) showUsageSpecify(CommonParams.MAX_NUM_RESULTS_DESC);
      
      // mMaxNumRet must be init before mMaxCandRet
      mMaxNumRet = Integer.MIN_VALUE;
      for (String s: mSplitOnComma.split(tmpn)) {
        int n = 0;
        
        try {
          n = Integer.parseInt(s);
        } catch (NumberFormatException e) {
          showUsage("Number of candidates isn't integer: '" + s + "'");
        }
  
        if (n <= 0) {
          showUsage("Specify only positive number of candidate entries");
        }
        
        mNumRetArr.add(n);      
        mMaxNumRet = Math.max(n, mMaxNumRet);
      }
    }
    
    {
      String tmpn = mCmd.getOptionValue(CommonParams.MAX_FINAL_RERANK_QTY_PARAM);
      if (tmpn != null) {
        try {
          mMaxFinalRerankQty = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Maximum number of entries to re-rank (using a final re-ranker) isn't integer: '" + tmpn + "'");
        } 
      }
    }
    
    if (mResultCacheName != null) logger.info("Cache file name: " + mResultCacheName);

    mMaxCandRet = mMaxNumRet; // if the user doesn't specify the # of candidates, it's set to the maximum # of answers to produce
    
    logger.info(
        String.format(
            "Candidate provider type: %s URI: %s Query file: %s Max. # of queries: %d # of cand. records: %d Max. # to re-rank w/ final re-ranker: %d", 
        mCandProviderType, mProviderURI, mQueryFile, mMaxNumQuery, mMaxCandRet, mMaxFinalRerankQty));
    mResultCacheName = mCmd.getOptionValue(CommonParams.QUERY_CACHE_FILE_PARAM);
    logger.info("An array of number of entries to retrieve:");
    for (int topk : mNumRetArr) {
      logger.info("" + topk);
    }
    
    mQrelFile = mCmd.getOptionValue(CommonParams.QREL_FILE_PARAM);
    if (mQrelFile != null) {
      logger.info("QREL-file: " + mQrelFile);
      mQrels = new QrelReader(mQrelFile);
    }
    {
      String tmpn = mCmd.getOptionValue(CommonParams.THREAD_QTY_PARAM);
      if (null != tmpn) {
        try {
          mThreadQty = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          showUsage("Number of threads isn't integer: '" + tmpn + "'");
        }
      }
    }
    logger.info(String.format("Number of threads: %d", mThreadQty));

    mGizaRootDir = mCmd.getOptionValue(CommonParams.GIZA_ROOT_DIR_PARAM);
    mEmbedDir = mCmd.getOptionValue(CommonParams.EMBED_DIR_PARAM);
    
    mUseThreadPool = mCmd.hasOption(CommonParams.USE_THREAD_POOL_PARAM);

    mFwdIndexPref = mCmd.getOptionValue(CommonParams.FWDINDEX_PARAM);
    mExtrTypeInterm = mCmd.getOptionValue(CommonParams.EXTRACTOR_TYPE_INTERM_PARAM);
    if (mExtrTypeInterm != null) {
      String modelFile = mCmd.getOptionValue(CommonParams.MODEL_FILE_INTERM_PARAM);
      if (null == modelFile) 
        showUsageSpecify(CommonParams.MODEL_FILE_INTERM_PARAM);
      mMaxCandRet = mMaxNumRet; // if the user doesn't specify the # of candidates, it's set to the maximum # of answers to produce
      mModelInterm = FeatureExtractor.readFeatureWeights(modelFile);
      {
        String tmpn = mCmd.getOptionValue(CommonParams.MAX_CAND_QTY_PARAM);
        if (null == tmpn)
          showUsageSpecify(CommonParams.MAX_CAND_QTY_DESC);
        try {
          mMaxCandRet = Integer.parseInt(tmpn);
          if (mMaxCandRet < mMaxNumRet)
            mMaxCandRet = mMaxNumRet; // The number of candidate records can't be < the the # of records we need to retrieve
        } catch (NumberFormatException e) {
          showUsage("The value of '" + CommonParams.MAX_CAND_QTY_DESC + "' isn't integer: '" + tmpn + "'");
        }
      }

      logger.info("Using the following weights for the intermediate re-ranker:");
      logger.info(mModelInterm.toString());
    }
    mExtrTypeFinal = mCmd.getOptionValue(CommonParams.EXTRACTOR_TYPE_FINAL_PARAM);
    if (mExtrTypeFinal != null) {
      if (mUseFinalModel) {
        String modelFile = mCmd.getOptionValue(CommonParams.MODEL_FILE_FINAL_PARAM);
        if (null == modelFile) 
          showUsageSpecify(CommonParams.MODEL_FILE_FINAL_PARAM);
        RankerFactory rf = new RankerFactory();
        File tmp = new File(modelFile);
        if (!tmp.exists()) {
          throw new Exception(String.format("Model file does not exist: %s", modelFile));
        }
        mModelFinal = rf.loadRankerFromFile(modelFile);
        logger.info("Loaded the final-stage model from the following file: '" + modelFile + "'");
      }
    }

    mSaveStatFile = mCmd.getOptionValue(CommonParams.SAVE_STAT_FILE_PARAM);
    if (mSaveStatFile != null)
      logger.info("Saving some vital stats to '" + mSaveStatFile);

  }
  
  /**
   * This function initializes feature extractors.
   * @throws Exception 
   */
  void initExtractors() throws Exception {
    mResourceManager = new FeatExtrResourceManager(mFwdIndexPref, mGizaRootDir, mEmbedDir);

    if (mExtrTypeFinal != null)
      mExtrFinal = new CompositeFeatureExtractor(mResourceManager, mExtrTypeFinal);
    if (mExtrTypeInterm != null)
      mExtrInterm = new CompositeFeatureExtractor(mResourceManager, mExtrTypeInterm);

  }
  
  /**
   * This function initializes the provider candidate provider. 
   * It should be called after {@link #initExtractors()}, which will create
   * an appropriate resource manager.
   * 
   * @throws Exception
   */
  void initProvider() throws Exception {
    mCandProviders = CandidateProvider.createCandProviders(mResourceManager, 
    														mCandProviderType, 
    														mProviderURI, 
    														mCandProviderConfigName, mThreadQty);    
    if (mCandProviders == null) {
      showUsage("Wrong candidate record provider type: '" + mCandProviderType + "'");
    }
  }

  
  /**
   * For debugging only.
   */
  static void printFloatArr(float a[]) {
    for (float v: a)  System.out.print(v + " ");
    System.out.println();
  }
  
  void run(String appName, String args[]) throws Exception {
    mAppName = appName;
    logger.info(appName + " started");
    
    try {
      long start = System.currentTimeMillis();
      
      // Read options
      addOptions();
      parseAndReadOpts(args);
      procCustomOptions();
      // Init. resources
      initExtractors();
      // Note that providers must be initialized after resources and extractors are initialized
      // because they may some of the resources (e.g., NMSLIB needs an in-memory feature extractor)
      initProvider();
      
      if (mResultCacheName != null) {
        mResultCache = new CandidateInfoCache();
        // If the cache file name is specified and it exists, read the cache!
        if (CandidateInfoCache.cacheExists(mResultCacheName)) { 
          mResultCache.readCache(mResultCacheName);
          logger.info("Result cache is loaded from '" + mResultCacheName + "'");
        }
      }      
      
      int queryQty = 0;
      
      try (DataEntryReader inp = new DataEntryReader(mQueryFile)) {
        Map<String, String> queryFields = null;      
        
        for (; ((queryFields = inp.readNext()) != null) && queryQty < mMaxNumQuery; ) {
           
          mParsedQueries.add(queryFields);
          ++queryQty;
          if (queryQty % 1000 == 0) logger.info("Read " + queryQty + " documents from " + mQueryFile);
        }
      }
      
      logger.info("Read " + queryQty + " documents from " + mQueryFile);
      
      init();
          
      if (mUseThreadPool ) {
        ExecutorService executor = Executors.newFixedThreadPool(mThreadQty);
        logger.info(String.format("Created a fixed thread pool with %d threads", mThreadQty));
        
        for (int iq = 0; iq < mParsedQueries.size(); ++iq) {
          executor.execute(new BaseQueryAppProcessingWorker(this, iq));
        }                  
       
        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
      } else {
        BaseQueryAppProcessingThread[] workers = new BaseQueryAppProcessingThread[mThreadQty];
        
        for (int threadId = 0; threadId < mThreadQty; ++threadId) {               
          workers[threadId] = new BaseQueryAppProcessingThread(this, threadId, mThreadQty);
        }
              
        // Start threads
        for (BaseQueryAppProcessingThread e : workers) e.start();
        // Wait till they finish
        for (BaseQueryAppProcessingThread e : workers) e.join(0);        
      }
      
      fin();
      
      long end = System.currentTimeMillis();
      double totalTimeMS = end - start;
      
      double queryTime = mQueryTimeStat.getMean(), intermRerankTime = 0, finalRerankTime = 0;
      
      logger.info(String.format("Query time (ms):             mean=%f std=%f", 
                                mQueryTimeStat.getMean(), mQueryTimeStat.getStandardDeviation()));
      logger.info(String.format("Number of entries found:     mean=%f std=%f",
          mNumRetStat.getMean(), mNumRetStat.getStandardDeviation()));

      if (mModelInterm != null) {
        logger.info(String.format("Interm. reranking time (ms): mean=%f std=%f", 
              mIntermRerankTimeStat.getMean(), mIntermRerankTimeStat.getStandardDeviation()));
        intermRerankTime = mIntermRerankTimeStat.getMean();
      }
      if (mModelFinal != null) {
        logger.info(String.format("Final  reranking time (ms): mean=%f std=%f", 
              mFinalRerankTimeStat.getMean(), mFinalRerankTimeStat.getStandardDeviation()));
        finalRerankTime = mFinalRerankTimeStat.getMean();
      }
      
      if (mSaveStatFile != null) {
        FileWriter f = new FileWriter(new File(mSaveStatFile));
        f.write("QueryTime\tIntermRerankTime\tFinalRerankTime\tTotalTime\n");
        f.write(String.format("%f\t%f\t%f\t%f\n", queryTime, intermRerankTime, finalRerankTime, totalTimeMS));
        f.close();
      }
      
      // Overwrite cache only if it doesn't exist or is incomplete
      if (mResultCacheName != null && !CandidateInfoCache.cacheExists(mResultCacheName)) {
        mResultCache.writeCache(mResultCacheName);
        logger.info("Result cache is loaded from '" + mResultCacheName + "'");        
      }
      
    } catch (ParseException e) {
      showUsageSpecify("Cannot parse arguments: " + e);
    } catch(Exception e) {
      e.printStackTrace();
      logger.error("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
    logger.info("Finished successfully!");
  }
  
  String       mRunId;
  String       mProviderURI;
  String       mQueryFile;
  Integer      mMaxCandRet;
  int          mMaxFinalRerankQty = Integer.MAX_VALUE;
  Integer      mMaxNumRet;
  int          mMaxNumQuery = Integer.MAX_VALUE;
  ArrayList<Integer> mNumRetArr= new ArrayList<Integer>();
  String       mCandProviderType;
  String       mCandProviderConfigName;
  String       mQrelFile;
  QrelReader   mQrels;
  int          mThreadQty = 1;
  String       mSaveStatFile;     
  String       mGizaRootDir;
  String       mEmbedDir;
  String       mFwdIndexPref;
  String       mExtrTypeFinal;
  String       mExtrTypeInterm;
  DenseVector  mModelInterm;
  Ranker       mModelFinal;
  boolean      mKnnInterleave = false;
  boolean      mUseThreadPool = false;
  FeatExtrResourceManager mResourceManager;
  
  String             mResultCacheName = null; 
  CandidateInfoCache mResultCache = null;
  
  FeatureExtractor mExtrInterm;
  FeatureExtractor mExtrFinal;
  
  String   mAppName;
  Options  mOptions = new Options();
  Splitter mSplitOnComma = Splitter.on(',');   
  
  CommandLineParser mParser = new org.apache.commons.cli.GnuParser();
  
  boolean  mMultNumRetr;    /** if true an application generate results for top-K sets of various sizes */
  boolean  mOnlyLucene;     /** if true, we don't allow to specify the provider type and only uses Lucene */
  boolean  mUseQRELs;       /** if true, a QREL file is read */
  boolean  mUseIntermModel; /** if true, an intermediate re-ranking model is used */
  boolean  mUseFinalModel;  /** if true, a final re-ranking model is used */
                                 
  CommandLine mCmd;
  
  CandidateProvider[] mCandProviders;
  
  SynchronizedSummaryStatistics mQueryTimeStat        = new SynchronizedSummaryStatistics();
  SynchronizedSummaryStatistics mIntermRerankTimeStat = new SynchronizedSummaryStatistics();
  SynchronizedSummaryStatistics mFinalRerankTimeStat  = new SynchronizedSummaryStatistics();
  SynchronizedSummaryStatistics mNumRetStat           = new SynchronizedSummaryStatistics();
  
  ArrayList<Map<String, String>> mParsedQueries = new ArrayList<Map<String, String>>();      
}
