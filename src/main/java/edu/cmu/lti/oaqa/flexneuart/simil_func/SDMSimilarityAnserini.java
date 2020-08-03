/**
 * A sequential dependency model ported from the Anserini toolkit:
 * https://github.com/castorini/anserini
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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;

public class SDMSimilarityAnserini implements QueryDocSimilarityFunc {
  
  private final float mLambdaT; // reasonable default 0.5f;
  private final float mLambdaO; // reasonable default 0.2f;
  private final float mLambdaU; // reasonable default  0.3f;
  private final int mWindowSize; // reasonable default 8;

  public SDMSimilarityAnserini(float lambdaT, float lambdaO, float lambdaU, int windowSize) {    
    mLambdaT = lambdaT;
    mLambdaO = lambdaO;
    mLambdaU = lambdaU;
    mWindowSize = windowSize;
  }
  
  private float computeUnorderedFrequencyScore(DocEntryParsed query, DocEntryParsed doc) {
    Map<Integer, Integer> queryPairMap = new HashMap<>();
    Map<Integer, Integer> phraseCountMap = new HashMap<>();
    Map<Integer, Integer> singleCountMap = new HashMap<>();
    
    int[] queryTokens = query.mWordIdSeq;
    // Construct a count map and a map of phrase pair x y, x->y
    for (int i = 0; i < queryTokens.length - 1; i++)  {
      queryPairMap.put(queryTokens[i], queryTokens[i+1]);
      phraseCountMap.put(queryTokens[i], 0);
      // This will serve as our smoothing param
      singleCountMap.put(queryTokens[i], 1);
    }
    
    int docSize = 0;
    int[] docTokens = doc.mWordIdSeq;
    // We will maintain a FIFO queue of window size
    LinkedList<Integer> window = new LinkedList<>();
    int docTokId = 0;
    for (; docTokId < docTokens.length && docSize <= mWindowSize * 2; docTokId++) {
      // First construct the window that we need to test on
      docSize ++;
      window.add(docTokens[docTokId]);
    }
    
    // Now we can construct counts for up to index mWindowSize -1
    // But we need to account for the case when the document just doesn't have that many tokens
    for (int i = 0; i < Math.min(mWindowSize - 1, docSize); i++) {
      int firstToken = window.get(i);
      if (queryPairMap.containsKey(firstToken) && window.contains(queryPairMap.get(firstToken))) {
        phraseCountMap.put(firstToken, phraseCountMap.get(firstToken) + 1);
        singleCountMap.put(firstToken, singleCountMap.get(firstToken) + 1);
      }
    }
    
    // Now we continue
    for (; docTokId < docTokens.length; docTokId++) {
      docSize ++;
      window.add(docTokens[docTokId]);
      // Move the window along
      // The window at this point is guaranteed to be of size mWindowSize * 2 because of the previous loop
      // if there are not enough tokens this would not even execute
      window.removeFirst();
      // Now test for the phrase at the test index mWindowSize -1
      int firstToken = window.get(mWindowSize-1);
      if (queryPairMap.containsKey(firstToken) && window.contains(queryPairMap.get(firstToken))) {
        phraseCountMap.put(firstToken, phraseCountMap.get(firstToken) + 1);
        singleCountMap.put(firstToken, singleCountMap.get(firstToken) + 1);
      }
    }
    
    float score = 0.0f;
    // Smoothing count of 1
    docSize ++;
    for (int queryToken : phraseCountMap.keySet()) {
      float countToUse = phraseCountMap.get(queryToken);
      if (countToUse == 0) {
        countToUse = singleCountMap.get(queryToken);
      }
      score += Math.log(countToUse/ (float) docSize);
    }

    return score;
    
  }
  
  private float computeOrderedFrequencyScore(DocEntryParsed query, DocEntryParsed doc) {
    Map<Integer, Integer> queryPairMap = new HashMap<>();
    Map<Integer, Integer> phraseCountMap = new HashMap<>();
    Map<Integer, Integer> singleCountMap = new HashMap<>();
    
    int[] queryTokens = query.mWordIdSeq;

    // Construct a count map and a map of phrase pair x y, x->y
    for (int i = 0; i < queryTokens.length - 1; i++)  {
      queryPairMap.put(queryTokens[i], queryTokens[i+1]);
      phraseCountMap.put(queryTokens[i], 0);
      // This will serve as our smoothing param
      singleCountMap.put(queryTokens[i], 1);
    }
    
    int[] docTokens = doc.mWordIdSeq; 
    float docSize = 0.0f;
    // Use these to track which token we need to see to increment count
    // count tracked on the first token
    int expectedToken = -2; // to make sure it isn't equal to an OOV token -1
    int tokenToIncrement = -2; // to make sure it isn't equal to an OOV token -1
    for (int docTokId = 0; docTokId < docTokens.length; docTokId++) {
      docSize++;
      int token = docTokens[docTokId];
      if (token == expectedToken) {
        phraseCountMap.put(tokenToIncrement, phraseCountMap.get(tokenToIncrement) + 1);
      }

      // Check now if this token could be the start of an ordered phrase
      if (queryPairMap.containsKey(token)) {
        expectedToken = queryPairMap.get(token);
        singleCountMap.put(token, singleCountMap.get(token) + 1);
        tokenToIncrement = token;
      } else {
        expectedToken = -2;
        tokenToIncrement = -2;
      }
    }
    float score = 0.0f;
    // Smoothing count of 1
    docSize ++;
    for (int queryToken : phraseCountMap.keySet()) {
      score += Math.log((float) (phraseCountMap.get(queryToken) + 1) / docSize);
    }

    return score;

  }
  
  private float computeFullIndependenceScore(DocEntryParsed query, DocEntryParsed doc) {
    // tf can be calculated by iterating over terms, number of times a term occurs in doc
    // |D| total number of terms can be calculated by iterating over stream
    Map<Integer, Integer> termCount = new HashMap<>();
    
    int[] docTokens = doc.mWordIdSeq; 
    
    float docSize =0;
    // Count all the tokens
    for (int docTokId = 0; docTokId < docTokens.length; docTokId++) {
      docSize++;
      int token = docTokens[docTokId];
      if (termCount.containsKey(token)) {
        termCount.put(token, termCount.get(token) + 1);
      } else {
        termCount.put(token, 1);
      }
    }
    
    int[] queryTokens = query.mWordIdSeq;
    
    float score = 0.0f;
    // Smoothing count of 1
    docSize ++;
    // Only compute the score for what's in term count all else 0
    for (int queryToken : queryTokens) {
      if (termCount.containsKey(queryToken)) {
        score += Math.log((float) (termCount.get(queryToken)  +1) / docSize);
      }
    }
    return score;    
  }
  
  /**
   * Computes the similarity between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  public float compute(DocEntryParsed query, DocEntryParsed doc) {
    float independentScore = computeFullIndependenceScore(query, doc);
    float orderedWindowScore = computeOrderedFrequencyScore(query, doc);
    float  unorderedDependenceScore = computeUnorderedFrequencyScore(query, doc);
    
    return mLambdaT * independentScore + mLambdaO * orderedWindowScore + mLambdaU * unorderedDependenceScore;
  }

}
