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

import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;

import com.google.common.base.Splitter;

public class LuceneCandidateProvider extends CandidateProvider {
  // 8 GB is quite reasonable, but if you increase this value, 
  // you may need to modify the following line in *.sh files
  // export MAVEN_OPTS="-Xms8192m -server"
  public static double RAM_BUFFER_SIZE = 1024 * 8; // 8 GB
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }  
  
  public LuceneCandidateProvider(String indexDirName,
                                 float k1, float b) throws Exception {
    File indexDir = new File(indexDirName);
    mSimilarity = new BM25Similarity(k1, b);
    
    if (!indexDir.exists()) {
      throw new Exception(String.format("Directory '%s' doesn't exist", indexDirName)); 
    }
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
    
    String query = StringUtilsLeo.removePunct(text.trim());

    ArrayList<String>   toks = new ArrayList<String>();
    for (String s: mSpaceSplit.split(query)) {
      toks.add(s);
    }
    if (2 * toks.size() > BooleanQuery.getMaxClauseCount()) {
      // This a heuristic, but it should work fine in many cases
      BooleanQuery.setMaxClauseCount(2 * toks.size());
    }

    long    numFound = 0;

    if (!query.isEmpty()) {    
      // QueryParser cannot be shared among threads!
      QueryParser parser = new QueryParser(TEXT_FIELD_NAME, mAnalyzer);
      parser.setDefaultOperator(QueryParser.OR_OPERATOR);

      Query       queryParsed = parser.parse(query);
      
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
      
    CandidateEntry[] results = resArr.toArray(new CandidateEntry[resArr.size()]);
    Arrays.sort(results);
        
    return new CandidateInfo(numFound, results);
  }
  
  private IndexReader   mReader = null;
  private IndexSearcher mSearcher = null;
  private final Similarity mSimilarity;
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();

  private static Splitter mSpaceSplit = Splitter.on(' ').omitEmptyStrings().trimResults();
}
