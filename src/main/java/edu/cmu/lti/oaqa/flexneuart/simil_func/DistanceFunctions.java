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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;

import no.uib.cipr.matrix.sparse.SparseVector;
import edu.cmu.lti.oaqa.flexneuart.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

/**
 * Implementations for some distance functions between vectors and/or strings.
 * 
 * @author Leonid Boytsov
 *
 */
public class DistanceFunctions {
  public static final float FLOAT_EPS = Float.MIN_NORMAL * 2;
  public static final int LCS_LIKE_QTY = 3;
  public static final int EMD_LIKE_QTY = 4;

  /**
   * Computes the Euclidean distance. 
   * Assumptions, vec1.length == vec2.length && vec1.length > 0
   * 
   * @param vec1    the first vector
   * @param vec2    the second vector
   * @return        normalized scalar product
   */
  public static float compEuclidean(float[] vec1, float[] vec2)  {    
    float sum = 0;
    
    int N = vec1.length;
    if (N != vec2.length) 
      throw new RuntimeException(String.format("Bug: different vector lengths: %d vs %d",
                                        vec1.length, vec2.length));
    if (N == 0)
      throw new RuntimeException("Bug: zero-length vectors are not acceptable!");

    for (int i = 0; i < N; i++) {
      float d = vec1[i] - vec2[i];
      sum += d * d;
    }

    return (float) (Math.sqrt(sum));    
  }
  
  /**
   * Computes the cosine distance. 
   * Assumptions, vec1.length == vec2.length && vec1.length > 0
   * 
   * @param vec1    the first vector
   * @param vec2    the second vector
   * @return        cosine distance
   */  
  public static float compCosine(float[] vec1, float[] vec2)  {
    return 1 - compNormScalar(vec1, vec2);
  }
  
  /**
   * Computes an unnormalized scalar product.
   * 
   * @param vec1    the first vector
   * @param vec2    the second vector
   * @return        a regular (unnormalized) inner/scalar product
   */
  public static float compScalar(float[] vec1, float[] vec2)  {    
    float sum = 0;

    int N = vec1.length;
    if (N != vec2.length) 
      throw new RuntimeException(String.format("Bug: different vector lengths: %d vs %d",
                                        vec1.length, vec2.length));
    if (N == 0)
      throw new RuntimeException("Bug: zero-length vectors are not acceptable!");

    for (int i = 0; i < N; i++) {
      sum += vec1[i] * vec2[i];
    }
    
    return sum;
  }
  
  /**
   * Computes a normalized scalar product, i.e., a cosine value
   * of the angle between vectors. 
   * Assumptions, vec1.length == vec2.length && vec1.length > 0
   * 
   * @param vec1    the first vector
   * @param vec2    the second vector
   * @return        normalized scalar product
   */
  public static float compNormScalar(float[] vec1, float[] vec2)  {    
    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    
    int N = vec1.length;
    if (N != vec2.length) 
      throw new RuntimeException(String.format("Bug: different vector lengths: %d vs %d",
                                        vec1.length, vec2.length));
    if (N == 0)
      throw new RuntimeException("Bug: zero-length vectors are not acceptable!");

    for (int i = 0; i < N; i++) {
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
      sum += vec1[i] * vec2[i];
    }

    /* 
     * Sometimes due to rounding errors, we get values > 1 or < -1.
     * This throws off other functions that use scalar product, e.g., acos
     */
    float normMul = norm1 * norm2;
    normMul =  (float) Math.sqrt(Math.max(FLOAT_EPS, normMul));

    float normSum = (float) (sum / normMul);

    float res = Math.max(-1f, Math.min(1.f, normSum));
    if (Float.isInfinite(res)) {
      throw new RuntimeException("Obtained inifinte normalized sum in the cosine distance computation!");
    }
    return res;
  }
  

  /**
   * Computes the longest common subsequence between two string sequences.
   * 
   * @param seq1    the first string.
   * @param seq2    the second string.
   * @return    the length of the longest common sequence.
   */
  public static int compLCS(String [] seq1, String [] seq2) {
    int len2 = seq2.length;
    int len1 = seq1.length;
    
    int[] colCurr = new int [len2 + 1];
    int[] colPrev = new int [len2 + 1];
    
    for (int i1 = 0; i1 < len1; i1++) {
      for (int i2 = 0; i2 < len2; i2++) {
        if (seq1[i1].equals(seq2[i2])) {
          colCurr[i2+1] = colPrev[i2] + 1;
        } else {
          colCurr[i2+1] = Math.max(colPrev[i2+1], colCurr[i2]);
        }
      }
    // Swap references 
      int[] tmp = colPrev;
      colPrev = colCurr;
      colCurr = tmp;      
    }
    
    return colPrev[len2];
  }

  /**
   * Computes the longest common subsequence between two integer sequences.
   * 
   * @param seq1    the first sequence.
   * @param seq2    the second sequence.
   * @return    the length of the longest common sequence.
   */
  public static int compLCS(int [] seq1, int [] seq2) {
    int len2 = seq2.length;
    int len1 = seq1.length;
    
    int[] colCurr = new int [len2 + 1];
    int[] colPrev = new int [len2 + 1];
    
    for (int i1 = 0; i1 < len1; i1++) {
      for (int i2 = 0; i2 < len2; i2++) {
        if (seq1[i1] == seq2[i2]) {
          colCurr[i2+1] = colPrev[i2] + 1;
        } else {
          colCurr[i2+1] = Math.max(colPrev[i2+1], colCurr[i2]);
        }
      }
    // Swap references 
      int[] tmp = colPrev;
      colPrev = colCurr;
      colCurr = tmp;      
    }
    
    return colPrev[len2];
  }
 
  /**
   * Computes a number terms shared between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  public static float compOverallMatch(DocEntryParsed query, DocEntryParsed doc) {
    float score = 0;
    
    int   docTermQty = doc.mWordIds.length;
    int   queryTermQty = query.mWordIds.length;
    
    int   iQuery = 0, iDoc = 0;       
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else {
        score +=  query.mQtys[iQuery];
        
        ++iQuery; ++iDoc;
      }
    }
    
    return score;
  }


  /**
   * Compute the metric in the spirit of the longest common subsequence between 
   * two string sequences while taking into account individual word similarity, 
   * which is expressed via the distance matrix, rather than merely exact equality. 
   * 
   * <p>Three version of the LCS are computed:</p>
   * <ol>
   * <li>The first one is similar to a regular
   * LCS in that words with a distance below a certain threshold are considered
   * equal.
   * <li>The second one also gives credit only to word pairs whose distance
   * is below a threshold. However, instead, of using the value of 1, as in the
   * case of exact equality, it used a fuzzy score from 0 to computed as 1 - distance.
   * Clearly, this makes sense only when the meaningful range of distances is from 0 to 1.
   * For example, this is the case of the cosine distance, or in the case
   * of the Euclidean distance between normalized vectors.
   * <li>The third one doesn't use a threshold, but merely uses the score 1- distance
   * as the fuzzy score. Again, this only makes sense when most distances
   * fall in the range [0,1].
   * </ol> 
   * 
   * @param distMatrix     a distance matrix, which CAN be empty; 
   *                       the matrix is represented by an array of array references,
   *                       where we assume that each referenced array has the same
   *                       number of elements.
   * @param distThresh     a distance threshold.
   * 
   * @return    a vector containing three LCS-inspired metrics described above. 
   */
  public static float[] compLCSLike(float distMatr[][],
                                    float distThresh) {
    if (distMatr.length == 0 || distMatr[0].length == 0) 
      return new float[LCS_LIKE_QTY];
    
    int len1 = distMatr.length;
    int len2 = distMatr[0].length;
    
    if (distThresh < 0) throw 
      new RuntimeException("The distance threshold should be non-negative!");
    
    float[] colCurrThresh = new float[len2 + 1];
    float[] colPrevThresh = new float[len2 + 1];
   
    float[] colCurrThreshFuzzy = new float[len2 + 1];
    float[] colPrevThreshFuzzy = new float[len2 + 1];
   
    float[] colCurrMaxSum = new float[len2 + 1];
    float[] colPrevMaxSum = new float[len2 + 1];
    
    for (int i1 = 0; i1 < len1; i1++) {
      for (int i2 = 0; i2 < len2; i2++) {        
        float  dist = distMatr[i1][i2];
        // The distance score is maximum when the distance is zero.
        float  distScore = Math.max(0.0f, 2.0f - dist) / 2.0f;
        
        float similThresh      = dist <= distThresh ? 1 : 0;
        float similThreshFuzzy = dist <= distThresh ? distScore : 0;
        
        colCurrThresh[i2+1]      = Math.max(Math.max(colPrevThresh[i2+1], colCurrThresh[i2]), 
                                                     colPrevThresh[i2] + similThresh);
        colCurrThreshFuzzy[i2+1] = Math.max(Math.max(colPrevThreshFuzzy[i2+1], colCurrThreshFuzzy[i2]), 
                                                     colPrevThreshFuzzy[i2] + similThreshFuzzy);
        colCurrMaxSum[i2+1]      = Math.max(Math.max(colPrevMaxSum[i2+1], colCurrMaxSum[i2]), 
                                                     colPrevMaxSum[i2] + distScore);
      }
     // Swap dynamic programming columns' references
      {
        float[] tmp = colPrevThresh;
        colPrevThresh = colCurrThresh;
        colCurrThresh = tmp;
      }
      
      {
        float[] tmp = colPrevThreshFuzzy;
        colPrevThreshFuzzy = colCurrThreshFuzzy;
        colCurrThreshFuzzy = tmp;
      }
                    
      {
        float[] tmp = colPrevMaxSum;
        colPrevMaxSum = colCurrMaxSum;
        colCurrMaxSum = tmp;
      }
     // End of reference swap                                       
    }
    
    float res[] = new float[LCS_LIKE_QTY];
    
    res[0]=colPrevThresh[len2];
    res[1]=colPrevThreshFuzzy[len2];
    res[2]=colPrevMaxSum[len2];
    
    return res;
  } 
  
  /**
   * Computes the distance matrix.
   * 
   * @param distType    an object encapsulating distance type.
   * @param vecs1        the first array of vector embeddings.
   * @param vecs2        the second array of vector embeddings.
   * 
   * @return the distance matrix: entries corresponding to null vectors
   *         are filled with Float.POSITIVE_INFINITY
   */
  
   public static float [][] compDistMatrix(AbstractDistance distType,
                              float[][] vecs1,
                              float[][] vecs2) {    
    int qty1 = vecs1.length, qty2 = vecs2.length;

    float distMatr[][] = new float[qty1][];
    
    for (int i = 0; i < qty1; ++i) {
      distMatr[i] = new float[qty2];
      
      for (int k = 0; k < qty2; ++k)
        distMatr[i][k] = (vecs1[i] != null && vecs2[k] != null) ? 
                            distType.compute(vecs1[i], vecs2[k]) : 
                            Float.POSITIVE_INFINITY;
    }
    
    return distMatr;
  }
  
  /**
   * Creates the distance matrix for given document entries:
   * word ID sequences are first mapped to embeddings.
   * 
   * @param distType    an object encapsulating distance type.
   * @param e1          the first document entry.
   * @param e2          the second document entry.
   * @param embed       an object that provides word embeddings.
   * 
   * @return the distance matrix: entries corresponding to null vectors
   *         are filled with Float.POSITIVE_INFINITY
   */
   public static float [][]  compDistMatrix(AbstractDistance distType,
                               DocEntryParsed e1, 
                               DocEntryParsed e2,
                               EmbeddingReaderAndRecoder embed) {
    int qty1 = e1.mWordIds.length;
    float vecs1 [][] = new float[qty1][];    
    for (int i = 0; i < qty1; ++i) {
      vecs1[i] = embed.getVector(e1.mWordIds[i]);
    }
    
    int qty2 = e2.mWordIds.length;
    float vecs2 [][] = new float[qty2][];    
    for (int i = 0; i < qty2; ++i) {
      vecs2[i] = embed.getVector(e2.mWordIds[i]);
    }   
    
    return compDistMatrix(distType, vecs1, vecs2);
  }

  /**
   * Creates the distance matrix for given word sequences:
   * sequences are first mapped to embeddings.
   * 
   * @param distType    an object encapsulating distance type.
   * @param words1      the first sequence of words.
   * @param e2          the second sequence of words.
   * @param embed       an object that provides word embeddings.
   * 
   * @return the distance matrix: entries corresponding to null vectors
   *         are filled with Float.POSITIVE_INFINITY
   */
  public static float [][]  compDistMatrix(AbstractDistance distType,
                               String   words1[],
                               String   words2[],
                               EmbeddingReaderAndRecoder embed) {
    int qty1 = words1.length;
    float vecs1 [][] = new float[qty1][];    
    for (int i = 0; i < qty1; ++i) {
      vecs1[i] = embed.getVector(words1[i]);
    }
    
    int qty2 = words2.length;
    float vecs2 [][] = new float[qty2][];    
    for (int i = 0; i < qty2; ++i) {
      vecs2[i] = embed.getVector(words2[i]);
    }   
    
    return compDistMatrix(distType, vecs1, vecs2);
  }  
  
  /**
   * Computes similarity scores related Word-Moving-Distance between two document entries based.
   * 
   * <p>Based on "From Word Embeddings To Document Distances" by Kusner et al 2015.
   *
   * @param e1          the first document entry.
   * @param e2          the second document entry.
   *
   * @param distMatrix     a distance matrix, which is assumed to be non-empty; 
   *                       the matrix is represented by an array of array references,
   *                       where we assume that each referenced array has the same
   *                       number of elements. The number of rows and columns should
   *                       be equal to the number of unique words in e1 and e2, 
   *                       respectively.
   *  
   * @return an array of four floats, where the first entry is the WMD-lower bound proposed
   *         by Kusner et al, other three entries are its different
   *         normalized and weighted versions (proposed by us).
   */
  public static float[] compEMDLike(DocEntryParsed e1, DocEntryParsed e2,
                                    float distMatr[][]) {
    int qty1 = distMatr.length;

    float minVals1[] = new float[qty1];
    int qty2 = distMatr[0].length;
    float minVals2[] = new float[qty2];
    
    Arrays.fill(minVals1, Float.POSITIVE_INFINITY);
    Arrays.fill(minVals2, Float.POSITIVE_INFINITY);
    
    for (int k1 = 0; k1 < qty1; ++k1) {
      for (int k2 = 0; k2 < qty2; ++k2) {
        float dist = distMatr[k1][k2];
        
        minVals1[k1] = Math.min(minVals1[k1], dist);
        minVals2[k2] = Math.min(minVals2[k2], dist);
      }
    }
     
    float distSum1 = 0, distSum2 = 0;
    
    for (int i = 0; i < qty1; ++i) {
      float v = minVals1[i];
      if (!Float.isInfinite(v))
        distSum1 += v * e1.mQtys[i];
    }
    
    for (int i = 0; i < qty2; ++i) {
      float v = minVals2[i];
      if (!Float.isInfinite(v))
        distSum2 += v * e2.mQtys[i];        
    }
    
    float [] res = new float[EMD_LIKE_QTY];
    
    res[0] = Math.max(distSum1, distSum2);
    
    if (qty1 > 0 && qty2 > 0) {
      res[1] = Math.max(distSum1/qty1, distSum2/qty2);
      res[2] = (distSum1/qty1 + distSum2/qty2)/2;
      res[3] = (distSum1 * qty1 + distSum2 * qty2)/(qty1 + qty2);
    }
    
    return res;
  }
  
  /**
   * Computes a Jensen-Shannon Divergence between two sparse probability vectors.
   * 
   * @param vec1 the first vector
   * @param vec2 the second value
   * @return the value of Jensen-Shannon Divergence.
   */
  public static double computeJSDiv(SparseVector vec1, SparseVector vec2) throws Exception {
    int qty1 = vec1.getUsed();
    int qty2 = vec2.getUsed();
    
    float res = 0;
    
    int i1 = 0, i2 = 0;
    
    int    ids1[]  = vec1.getIndex();
    double vals1[] = vec1.getData();
    
    int    ids2[]  = vec2.getIndex();
    double vals2[] = vec2.getData();
    
    while (i1 < qty1 && i2 < qty2) {
      double val1 = 0, val2 = 0;
      if (ids1[i1] < ids2[i2]) {
        val1 = vals1[i1];
        if (val1 < 0 || val1 > 1) 
          throw new Exception(String.format("Illegal probability value %f", val1));
        i1++;
      } else if (ids1[i1] > ids2[i2]) {
        val2 = vals2[i2];
        if (val2 < 0 || val2 > 1) 
          throw new Exception(String.format("Illegal probability value %f", val2));        
        i2++;
      } else {
        val1 = vals1[i1];
        val2 = vals2[i2];
        i1++;
        i2++;
      }
      double valM = 0.5 * (val1+val2);
      if (Math.min(val1, valM) > Double.MIN_VALUE) {
        res += val1*Math.log(val1/valM);
      }
      if (Math.min(val2, valM) > Double.MIN_VALUE) {
        res += val2*Math.log(val2/valM);
      }      
    }
    while (i1 < qty1) {
      double val1 = vals1[i1];
      double valM = 0.5* val1;
      if (Math.min(val1, valM) > Double.MIN_VALUE) {
        res += val1*Math.log(val1/valM);
      }
      i1++;
    }
    while (i2 < qty2) {
      double val2 = vals2[i2];
      double valM = 0.5* val2;
      if (Math.min(val2, valM) > Double.MIN_VALUE) {
        res += val2*Math.log(val2/valM);
      }
      i2++;
    }    
    
    return res / 2;
  }
  
  public static void main(String[] arg) throws Exception {
    EmbeddingReaderAndRecoder wr = new EmbeddingReaderAndRecoder(arg[0], null);
    
    BufferedReader sysInReader = new BufferedReader(new InputStreamReader(System.in));

    while (true) {
      String word1 = null, word2 = null;
      
      System.out.println("Input 1st line of space-separated words: ");
      word1 = sysInReader.readLine();
      System.out.println("Input 2d line of space-separted words: ");
      word2 = sysInReader.readLine();
      float thresh = 0;
      System.out.println("Input the distance threshold (<1): ");
      String tmp = sysInReader.readLine();
      thresh = Float.parseFloat(tmp);
      
      String[] seq1=StringUtils.splitNoEmpty(word1, "\\s+");
      String[] seq2=word2.split("\\s+");
      
      float [][] distMatr = compDistMatrix(AbstractDistance.create("cosine"),
                                             seq1, seq2, wr);
      
      System.out.println("Distance matrix: ");
      
      for (int i = 0; i < distMatr.length; ++i) {
        for (int k = 0; k < distMatr[i].length; ++k) {
          System.out.print(
              String.format("d(%s,%s)=%f ", seq1[i], seq2[k], distMatr[i][k]));
        }
        System.out.println();
      }
      
      
      float res[] = compLCSLike(distMatr, thresh); 
      System.out.println("Thresholded LCS:         " + res[0]);
      System.out.println("Thresholded LCS (fuzzy): " + res[1]);
      System.out.println("Subsequence sum : " + res[2]);
    }
  }
}
