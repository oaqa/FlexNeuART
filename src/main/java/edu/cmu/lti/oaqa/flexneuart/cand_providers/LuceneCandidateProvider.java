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
package edu.cmu.lti.oaqa.flexneuart.cand_providers;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.simil_func.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

import com.google.common.base.Splitter;

public class LuceneCandidateProvider extends CandidateProvider {
	public static final String B_PARAM = "b";
	public static final String K1_PARAM = "k1";

	final Logger logger = LoggerFactory.getLogger(LuceneCandidateProvider.class);
	
  // 8 GB is quite reasonable, but if you increase this value, 
  // you may need to modify the following line in *.sh files
  // export MAVEN_OPTS="-Xms8192m -server"
  public static double DEFAULT_RAM_BUFFER_SIZE = 1024 * 8; // 8 GB
  
  @Override
  public String getName() {
    return this.getClass().getName();
  }  
  
  /**
   * Constructor.
   * 
   * @param indexDirName    Lucene index directory
   * @param addConf         additional/optional configuration: can be null
   * @throws Exception
   */
  public LuceneCandidateProvider(String indexDirName, CandProvAddConfig addConf) throws Exception {
  	
  	float k1 = BM25SimilarityLucene.DEFAULT_BM25_K1;
  	float b = BM25SimilarityLucene.DEFAULT_BM25_B;
  	
  	if (addConf != null) {
    	k1 = addConf.getParam(K1_PARAM, k1);
    	b = addConf.getParam(B_PARAM, b);
  	}
  	
  	logger.info(String.format("Lucene candidate provider %s=%g, %s=%g", K1_PARAM, k1, B_PARAM, b));
  	
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
    
    String text = queryData.get(QUERY_FIELD_NAME);
    if (null == text) {
      throw new Exception(
          String.format("Query (%s) is undefined for query # %d",
                        QUERY_FIELD_NAME, queryNum));
    }
    
    String query = StringUtils.removePunct(text.trim());

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
      QueryParser parser = new QueryParser(QUERY_FIELD_NAME, mAnalyzer);
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
