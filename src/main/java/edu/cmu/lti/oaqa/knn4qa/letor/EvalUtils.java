/*
 *  Copyright 2017 Carnegie Mellon University
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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;

public class EvalUtils {
  protected static final String NL = System.getProperty("line.separator");
  
  /** Some fake document ID, which is unlikely to be equal to a real one */
  private static final String FAKE_DOC_ID = 
      "THIS_IS_A_VERY_LONG_FAKE_DOCUMENT_ID_THAT_SHOULD_NOT_MATCH_ANY_REAL_ONES";

  /**
   * Saves query results to a TREC result file.
   * 
   * @param topicId     a question ID.
   * @param results     found entries to memorize.
   * @param trecFile    an object used to write to the output file.
   * @param runId       a run ID.
   * @param maxNum      a maximum number of results to save (can be less than 
   *                    the number of retrieved entries).
   * @throws IOException
   */
  public static void saveTrecResults(
                              String           topicId,
                              CandidateEntry[] results,
                              BufferedWriter   trecFile,
                              String           runId,
                              int maxNum) throws IOException {
    boolean bNothing = true;
    for (int i = 0; i < Math.min(results.length, maxNum); ++i) {
      bNothing = false;
      saveTrecOneEntry(trecFile, 
                       topicId, results[i].mDocId, 
                       (i+1), results[i].mScore, runId);
    }
    /*
     *  If nothing is returned, let's a fake entry, otherwise trec_eval
     *  will completely ignore output for this query (it won't give us zero
     *  as it should have been!)
     */
    if (bNothing) {
      saveTrecOneEntry(trecFile, 
          topicId, FAKE_DOC_ID, 
          1, 0, runId);      
    }
  }
  
  
  /**
   * Save positions, scores, etc information for a single retrieved documents.
   * 
   * @param trecFile    an object used to write to the output file.
   * @param topicId     a question ID.
   * @param docId       a document ID of the retrieved document.
   * @param docPos      a position in the result set (the smaller the better).
   * @param score       a score of the document in the result set.
   * @param runId       a run ID.
   * @throws IOException
   */
  private static void saveTrecOneEntry(BufferedWriter trecFile,
                                       String         topicId,
                                       String         docId,
                                       int            docPos,
                                       float          score,
                                       String         runId
                                       ) throws IOException {
    trecFile.write(String.format("%s\tQ0\t%s\t%d\t%f\t%s%s",
        topicId, docId, 
        docPos, score, runId,
        NL));    
  }
  
  /**
   * Read TREC run files, but ignore run ID(s).
   * 
   * @param fileName    input file (can be gz or bz2 compressed).
   * @return A hash map where keys are topic IDs and values are arrays. Each array represents a list of document for a single topic/query.
   * 
   * @throws Exception
   */
  public static HashMap<String, ArrayList<CandidateEntry>> readTrecResults(String fileName) throws Exception {
    HashMap<String, ArrayList<CandidateEntry>> res = new HashMap<String, ArrayList<CandidateEntry>>();
    
    BufferedReader inp = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fileName)));
    
    String line = null;
    
    String prevTopicId = null;
    int prevDocPos = -1;
    
    for (int lineNum = 1; (line = inp.readLine()) != null; ++lineNum) {
      line = line.trim();
      if (line.isEmpty()) continue;
      String parts[] = line.split("\\s+");
      if (parts.length != 6)
        throw new Exception(
              String.format("Wrong number of fields %d (expected 6) in line %d, file %s", 
                            parts.length, lineNum, fileName));
      String topicId = parts[0], docId = parts[2], docPosStr = parts[3], scoreStr = parts[4];
      
      float score = 0;
      int   docPos = -1;
      
      try {
        score = Float.parseFloat(scoreStr);
      } catch (NumberFormatException e) {
        throw new Exception(
            String.format("Error converting score in line %d, file %s", 
                          lineNum, fileName));
      }
      try {
        docPos = Integer.parseInt(docPosStr);
      } catch (NumberFormatException e) {
        throw new Exception(
            String.format("Error converting position in line %d, file %s", 
                          lineNum, fileName));
      }
      
      if (prevTopicId == null || !topicId.equals(prevTopicId)) {
        if (res.containsKey(topicId)) {
          throw new Exception(
              String.format("Duplicate topic %s in line %d, file %s", 
                            topicId, lineNum, fileName));          
        }
        res.put(topicId, new ArrayList<CandidateEntry>());
        prevDocPos = 0;
      }
      
      ArrayList<CandidateEntry> currArr = res.get(topicId);
      
      if (docPos != prevDocPos + 1) {
        throw new Exception(
            String.format("prevDocPos + 1 (%d) != docPos (%d) in line %d, file %s", 
                          prevDocPos + 1, docPos, lineNum, fileName));                
      }
      
      currArr.add(new CandidateEntry(docId, score));
       
      prevTopicId = topicId;
      prevDocPos = docPos;
    }
    
    return res;    
  }
  

  /**
   * Add one line to the TREC QREL file. 
   * 
   * @param qrelFile
   * @param topicId
   * @param docId
   * @param relGrade
   */
  public static void saveQrelOneEntry(BufferedWriter qrelFile,
                                      String           topicId,
                                      String           docId,
                                      int              relGrade) throws IOException {
    qrelFile.write(String.format("%s 0 %s %d%s", topicId, docId, relGrade, NL));
  }
  
  /**
   * Reading TREC QREL entries.
   * 
   * @param fileName   input file (can be gz or bz2 compressed).
   * @return a hash map of hash maps, where the outer hash map key is topic ID and the inner hash map key
   *         is a document id, inner hash values are relevance grades (integers).
   * @throws Exception
   */
  public static HashMap<String,HashMap<String, Integer>> readQrelEntries(String fileName) throws Exception {
    HashMap<String,HashMap<String, Integer>> res = new HashMap<String,HashMap<String, Integer>>();
    
    BufferedReader inp = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fileName)));
    
    String line = null;

    for (int lineNum = 1; (line = inp.readLine()) != null; ++lineNum) {
      line = line.trim();
      if (line.isEmpty()) continue;
      String parts[] = line.split("\\s+");
      if (parts.length != 4)
        throw new Exception(
              String.format("Wrong number of fields %d (expected 4) in line %d, file %s", 
                            parts.length, lineNum, fileName));
      String topicId = parts[0], docId = parts[2], relGradeStr = parts[3];
      
      int relGrade = 0;
      
      try {
        relGrade = Integer.parseInt(relGradeStr);
      } catch (NumberFormatException e) {
        throw new Exception(
            String.format("Error converting integer relevance grade in line %d, file %s", 
                          lineNum, fileName));
      }
      HashMap<String, Integer> topicRels = res.get(topicId);
      if (topicRels == null) {
        res.put(topicId, new HashMap<String, Integer>());
        topicRels = res.get(topicId);        
      }
      if (topicRels.containsKey(docId)) {
        throw new Exception(
            String.format("Duplicate docId (%s) or topicId (5s) in line %d, file %s", 
                          docId, topicId, lineNum, fileName));
        
      }
      topicRels.put(docId, relGrade);
    }
    
    return res;
  }
}
