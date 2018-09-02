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

import no.uib.cipr.matrix.DenseVector;

import java.util.*;

import edu.cmu.lti.oaqa.knn4qa.letor.InMemIndexFeatureExtractorOld;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntryExt;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;

class SearchEntry implements Comparable<SearchEntry> {
  /**
   * An identifier, e.g., a document ID or a word. 
   */
  public final String    mID;
  /**
   * The distance (the larger is better).
   */
  public final float     mScore;  
  
  public SearchEntry(String id, float score) {
    this.mID = id;
    this.mScore = score;
  }

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "[Identifier: " + mID + ", score: " + mScore + "]";
  }

  /* (non-Javadoc)
   * The smallest value should go first, this is necessary
   * for the priority queue (where the head is the SMALLEST value), i.e.,
   * we want to evict the worst similarity entry from the priority queue.
   */
  @Override
  public int compareTo(SearchEntry o) {
    if (mScore < o.mScore) return -1;
    if (mScore > o.mScore) return 1;
    return 0;
  }  
}

class BruteForceKNNThread extends Thread {
  private final int                           mTopK;
  private final int                           mThreadId;
  private final InMemIndexFeatureExtractorOld    mFeatExtr;
  private final DenseVector                   mWeights;
  private final int                           mThreadQty;
  private final Map<String, String>           mQueryData;
  private SearchEntry[]                       mResult = null;

  public SearchEntry [] getResult() {
    return mResult;
  }
  
  public BruteForceKNNThread(Map<String, String> queryData,
                      int topK,
                      InMemIndexFeatureExtractorOld featExtr,
                      int threadId,
                      int threadQty,
                      DenseVector weights) {
    mQueryData = queryData;
    mTopK = topK;
    mThreadId = threadId;
    mThreadQty = threadQty;
    mFeatExtr = featExtr;
    mWeights = weights;
  }
  
  @Override
  public void run() {
    InMemForwardIndex       fieldIndex = mFeatExtr.getTextFieldIndex();
    ArrayList<DocEntryExt>  allDocs = fieldIndex.getDocEntries();

    PriorityQueue<SearchEntry> q = new PriorityQueue<SearchEntry>(mTopK);
    ArrayList<String>          oneDocId = new ArrayList<String>();
    oneDocId.add("");
    
    for (int i = 0; i < allDocs.size(); ++i) { 
      if (i % mThreadQty == mThreadId) {
        DocEntryExt doc = allDocs.get(i);
        oneDocId.set(0, doc.mId);
        
        try {
          Map<String, DenseVector> res = mFeatExtr.getFeatures(oneDocId, mQueryData);
          
          DenseVector feat = res.get(doc.mId);
          float score = (float) feat.dot(mWeights);
          
          if (q.size() < mTopK) {
            q.add(new SearchEntry(doc.mId, score));
          } else if (score > q.peek().mScore) {
            q.add(new SearchEntry(doc.mId, score));
            q.poll();
          }
          
        } catch (Exception e) {
          e.printStackTrace();
          System.err.println("Search failure, exiting!");
          System.exit(1);
        }    
      }      
    }
    
    mResult = new SearchEntry[q.size()];
    mResult = q.toArray(mResult);
  }
                      
}

/**
 * Brute-force KNN to return k-closest entries.
 * 
 * @author Leonid Boytsov
 *
 */
public class BruteForceKNNCandidateProvider extends CandidateProvider {
  private InMemIndexFeatureExtractorOld    mFeatExtr;
  private DenseVector                   mWeights;
  private int                           mThreadQty;

  /**
   * @throws Exception 
   */
  public BruteForceKNNCandidateProvider(
      InMemIndexFeatureExtractorOld featExtr,
      DenseVector       weights,
      int               threadQty
      ) throws Exception {
    mFeatExtr = featExtr;
    mWeights   = weights;
    mThreadQty = Math.min(threadQty, 
                          mFeatExtr.getTextFieldIndex().getDocQty());
  }


  @Override
  public String getName() {
    return this.getClass().getName();
  }


  /*
   * getCandidates should be thread-safe, because the feature extractor's function
   * getFeatures is thread-safe. 
   */  
  @Override
  public boolean isThreadSafe() { return true; }   

  @Override
  public CandidateInfo getCandidates(int queryNum, Map<String, String> queryData,
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
    
    String query = text.trim();

    if (!query.isEmpty()) {
      
      
      BruteForceKNNThread  [] workers = new BruteForceKNNThread[mThreadQty];
      /*
       *  Create search threads: because the search is very slow (dozens of seconds
       *  it's quite cheap to create a thread for each request).
       */
      for (int threadId = 0; threadId < mThreadQty; ++threadId) {
        workers[threadId] = new BruteForceKNNThread(queryData, maxQty, mFeatExtr, 
                                            threadId, mThreadQty, mWeights);
      }
      // Start threads
      for (BruteForceKNNThread e : workers) e.start();
      // Wait till they finish
      for (BruteForceKNNThread e : workers) e.join(0);
      
      // Merge results
      for (BruteForceKNNThread thread : workers)
        for (SearchEntry e : thread.getResult()) {
          resArr.add(new CandidateEntry(e.mID, e.mScore));
        }
      
      Collections.sort(resArr);
      
      if (resArr.size() > maxQty) {
        ArrayList<CandidateEntry> tmp = resArr;
        resArr = new ArrayList<CandidateEntry>(maxQty); 
        for (int i = 0; i < maxQty; ++i)
          resArr.add(tmp.get(i));
      }
    }
    
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
//    Arrays.sort(results);
        
    return new CandidateInfo(results);
  }

}
