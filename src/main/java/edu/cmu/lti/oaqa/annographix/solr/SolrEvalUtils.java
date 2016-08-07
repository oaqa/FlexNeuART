/*
 *  Copyright 2014 Carnegie Mellon University

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
package edu.cmu.lti.oaqa.annographix.solr;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.File;
import java.util.regex.*;
import java.util.ArrayList;

import org.apache.commons.io.FileUtils;

public class SolrEvalUtils {
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
                              SolrRes[]        results,
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
  
  /*
   * This is for our private use only.
   */
  public static void saveEvalResults(
                              String            questionTemplate,
                              String            topicId,
                              SolrRes[]         results,
                              ArrayList<String> allKeyWords,
                              String            docDirName,
                              int maxNum) throws Exception {
    File    docRootDir = new File(docDirName);
    
    if (!docRootDir.exists()) {
      if (!docRootDir.mkdir()) 
        throw new Exception("Cannot create: " + docRootDir.getAbsolutePath());      
    }
    
    String  docDirPrefix = docDirName + "/" + topicId;
    File    docDir = new File(docDirPrefix);
    
    if (!docDir.exists()) {
      if (!docDir.mkdir()) 
        throw new Exception("Cannot create: " + docDir.getAbsolutePath());
    }
    
    // Let's precompile replacement regexps
    Pattern [] replReg  = new Pattern[allKeyWords.size()];
    String  [] replRepl = new String[allKeyWords.size()];
    
    for (int i = 0; i < allKeyWords.size(); ++i) {
      replReg[i] = Pattern.compile("(^| )(" + allKeyWords.get(i) + ")( |$)",
                                    Pattern.CASE_INSENSITIVE);
      replRepl[i] = "$1<b>$2</b>$3";
    }
    
    
    for (int docNum = 0; docNum < Math.min(results.length, maxNum); ++docNum) {
      String                docId   = results[docNum].mDocId;
      ArrayList<String>     docText = results[docNum].mDocText;
      StringBuilder         sb = new StringBuilder();
            
      sb.append("<html><body><br>");
      sb.append("<p><i>" + questionTemplate + "</i></p><hr><br>");
      
      for (String s: docText) {
        for (int k = 0; k < replReg.length; ++k) {
          while (true) {
            /*
             *  When words share a space, the replacement will not work in the first pass
             *  Imagine you have 
             *  word1 word2
             *  And both words need to be replaced. In the first pass, only 
             *  word1 is replaced. 
             */
            String sNew = replReg[k].matcher(s).replaceAll(replRepl[k]);
            if (sNew.equals(s)) break;
            s = sNew;
          }
        }

        sb.append(s);
        sb.append("<br>");
      }
      
      sb.append("</body></html>");
      
      String text = sb.toString();
    
      File  df = new File(docDirPrefix + "/" + docId + ".html");      

      // Don't overwrite docs!
      if (!df.exists()) {        
        try {
          FileUtils.write(df, text);
        } catch (IOException e) {
          throw new Exception("Cannot write to file: " + df.getAbsolutePath() 
              + " reason: " + e); 
        }
      } else {
        System.out.println(String.format(
            "WARNING: ignoring already created document for topic %s docId %s",
            topicId, docId));
      }
    }
  }

}
