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
package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.*;
import java.io.*;
import java.nio.file.Paths;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import edu.cmu.lti.oaqa.knn4qa.giza.GizaOneWordTranRecs;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.InMemIndexFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.WordEntry;

class TranRecSortByProb implements Comparable<TranRecSortByProb> {
  public TranRecSortByProb(int mDstWorId, float mProb) {
    super();
    this.mDstWorId = mDstWorId;
    this.mProb = mProb;
  }
  final public int     mDstWorId;
  final public float   mProb;
  @Override
  public int compareTo(TranRecSortByProb o) {
    // If mProb > o.mProb, we return -1
    // that is higher-probability entries will go first
    return (int) Math.signum(o.mProb - mProb);
  }
}

/**
 * A Lucene-based candidate provider that carries out a query expansion using
 * GIZA-computed translation probabilities: it works only with a <b>flipped/inverted</b>
 * translation table.
 * 
 * @author Leonid Boytsov
 *
 */
public class LuceneGIZACandidateProvider extends CandidateProvider {
  public static final String ERROR_EXPL = "The Lucene+GIZA candidate provider requires a Giza-enabled feature extractor";
  private static final Object lock = new Object();
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }  
  
  public LuceneGIZACandidateProvider(String indexDirName, int topTranQty, boolean useWeights,
                                    String gizaRootDir, int gizaIterQty, 
                                    String memIndexPref,
                                    InMemIndexFeatureExtractor ... featureExtractors) throws Exception {
    File indexDir = new File(indexDirName);
    boolean hasExtr = false;
    for (int k = 0; k < featureExtractors.length; ++k) {
      InMemIndexFeatureExtractor featExtr = featureExtractors[k];
      if (null != featExtr) {
        hasExtr = true;
        if (mAnswToQuestTran == null) mAnswToQuestTran = featExtr.getGizaTranTable(FeatureExtractor.TEXT_FIELD_ID);
        if (mFieldIndex == null)      mFieldIndex = featExtr.getFieldIndex(FeatureExtractor.TEXT_FIELD_ID);
      }
    }
    if (!hasExtr) {
      // This extractor name is hardcoded, but screw, this is good enough for a test!
      InMemIndexFeatureExtractor featExtr 
          = InMemIndexFeatureExtractor.createExtractor("exper@bm25=text+model1=text", 
                                                        gizaRootDir, gizaIterQty, 
                                                        memIndexPref, 
                                                        null, null, null);
      mAnswToQuestTran = featExtr.getGizaTranTable(FeatureExtractor.TEXT_FIELD_ID);
      mFieldIndex = featExtr.getFieldIndex(FeatureExtractor.TEXT_FIELD_ID);

    }
    if (mAnswToQuestTran == null) {
      throw new Exception("GIZA table for text field is null: " + ERROR_EXPL);
    }
    if (mFieldIndex == null) {
      throw new Exception("Field index for text field is null: " + ERROR_EXPL);
    }
    
    if (!indexDir.exists()) {
      throw new Exception(String.format("Directory '%s' doesn't exist", indexDirName)); 
    }
    mTopTranQty = topTranQty;
    mUseWeights = useWeights;
    
    mReader = DirectoryReader.open(FSDirectory.open(Paths.get(indexDirName)));
    mSearcher = new IndexSearcher(mReader);
    mSearcher.setSimilarity(mSimilarity);
    
  }
  
  /*
   *  The function getCandidates is thread-safe, because IndexSearcher is thread safe: 
   *  https://wiki.apache.org/lucene-java/LuceneFAQ#Is_the_IndexSearcher_thread-safe.3F
   */
  @Override
  public boolean isThreadSafe() { return true; }  
  
  @Override
  public CandidateInfo getCandidates(int queryNum, 
                                Map<String, String> queryData, 
                                int maxQty) throws Exception {
    ArrayList<CandidateEntry> resArr = new ArrayList<CandidateEntry>();
    
    String queryID = queryData.get(ID_FIELD_NAME);
    if (null == queryID) {
      throw new Exception(
          String.format("Query id (%s) is undefined for query # %d",
                        ID_FIELD_NAME, queryNum));
    }        
    
    String text = queryData.get(TEXT_FIELD_NAME);
    if (null == text) {
      throw new Exception(
          String.format("Query (%s) is undefined for query # %d",
                        TEXT_FIELD_NAME, queryNum));
    }
    
    String origQuery = text.trim();
    int    numFound = 0;

    if (!origQuery.isEmpty()) {    
      StringBuffer queryToks = new StringBuffer();
      int          tokQty = 0;
      
      for (String w : origQuery.split("\\s+")) 
      if (!w.isEmpty()) {
        tokQty++;        
        queryToks.append(w + " ");        
        
        final WordEntry we = mFieldIndex.getWordEntry(w);
        if (we != null) {
          GizaOneWordTranRecs rec0 = mAnswToQuestTran.getTranProbs(we.mWordId);
          if (rec0 != null) {
            TranRecSortByProb rec[] = new TranRecSortByProb[rec0.mDstIds.length];
            for (int i = 0; i < rec0.mDstIds.length; ++i)
              rec[i] = new TranRecSortByProb(rec0.mDstIds[i], rec0.mProbs[i]);
            Arrays.sort(rec);
            int qty = mTopTranQty;
            for (int i = 0; i < Math.min(rec.length, qty); ++i) {
              int dstId = rec[i].mDstWorId;
              if (dstId != we.mWordId) { 
                queryToks.append(mFieldIndex.getWord(dstId) + 
                    (mUseWeights ? "^" + String.format("%.3f", rec[i].mProb) : "") +" ");
                ++tokQty;
              } else {
                // If we skip a word, b/c it's the same as the query word
                // we will get one more candidate so that exactly Math.min(mTopTranQty, rec.length)
                // words were added
                ++qty;
              }              
            }
          }
        }
      }
      
      synchronized (lock) { // this is a static lock, it will block all instance of this class
        if (tokQty > BooleanQuery.getMaxClauseCount()) {
          BooleanQuery.setMaxClauseCount(tokQty);        
        }
      }
      
      String luceneQuery = queryToks.toString().trim();
      if (!luceneQuery.isEmpty()) {
        // QueryParser cannot be shared among threads!
        QueryParser parser = new QueryParser(TEXT_FIELD_NAME, mAnalyzer);
        parser.setDefaultOperator(QueryParser.OR_OPERATOR);
        
        Query       queryParsed = parser.parse(luceneQuery);
        
        //System.out.println("The resulting query: " + luceneQuery.toString());
        
        TopDocs     hits = mSearcher.search(queryParsed, maxQty);
        numFound = hits.totalHits;
        ScoreDoc[]  scoreDocs = hits.scoreDocs;
        
        for (ScoreDoc oneHit: scoreDocs) {
          Document doc = mSearcher.doc(oneHit.doc);
          String id = doc.get(ID_FIELD_NAME);
          float score = oneHit.score;
          
          resArr.add(new CandidateEntry(id, score));
        }
      }
    }
      
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
    Arrays.sort(results);
        
    return new CandidateInfo(numFound, results);
  }
  
  private IndexReader   mReader = null;
  private IndexSearcher mSearcher = null;
  private Similarity    mSimilarity = new BM25Similarity(FeatureExtractor.BM25_K1, FeatureExtractor.BM25_B);
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();
  private InMemForwardIndex             mFieldIndex = null;
  private GizaTranTableReaderAndRecoder mAnswToQuestTran = null;
  
  
  private final int                           mTopTranQty;
  private final boolean                       mUseWeights;
 }
