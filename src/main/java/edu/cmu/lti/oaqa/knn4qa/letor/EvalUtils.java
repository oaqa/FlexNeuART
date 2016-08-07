package edu.cmu.lti.oaqa.knn4qa.letor;

import java.io.BufferedWriter;
import java.io.IOException;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;

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
}
