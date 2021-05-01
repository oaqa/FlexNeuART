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
package edu.cmu.lti.oaqa.flexneuart.letor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;

public abstract class FeatureExtractor {  
  public abstract String getName();
  
  /**
   * Obtains features for a set of documents, this function should be <b>thread-safe!</b>.
   * 
   * @param     cands        an array of candidate entries.
   * @param     queryFields  a multi-field representation of the query {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}. 

   * @return a map docId -> sparse feature vector
   */
  public abstract Map<String,DenseVector> getFeatures(CandidateEntry[] cands, 
                                                      DataEntryFields  queryFields) throws Exception;
  
  /**
   * @return the total number of features (some may be missing, though).
   */
  public abstract int getFeatureQty();
   
  
  /**
   * Saves features (in the form of a sparse vector) to a file.
   * 
   * @param vect        feature weights to save
   * @param fileName    an output file
   */
  public static void saveFeatureWeights(DenseVector vect, String fileName) throws IOException {    
    BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(fileName)));
    StringBuffer sb = new StringBuffer();
       
    for (int i = 0; i < vect.size(); ++i)
      sb.append((i+1) + ":" + vect.get(i) + " ");
    
    outFile.write(sb.toString() + System.getProperty("line.separator"));
         
    outFile.close();    
  }
  
  /**
   * Reads feature weights from a file.
   * 
   * @param fileName    input file (in the RankLib format): all weights must be present
   *                    there should be no gaps!
   * @return            a sparse vector that keeps weights
   * @throws Exception
   */
  public static DenseVector readFeatureWeights(String fileName) throws Exception {
    BufferedReader inFile = new BufferedReader(new FileReader(new File(fileName)));
    
    
    try {
      String line = null;
      
      while ((line = inFile.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty() || line.startsWith("#")) continue;
        
        String parts0[] = line.split("\\s+");
        
        DenseVector res = new DenseVector(parts0.length);
        
        int ind = 0;
        for (String onePart: parts0) {
          try {
            String parts1[] = onePart.split(":");
            if (parts1.length != 2) {
              throw new Exception(
                  String.format(
                      "The line in file '%s' has a field '%s' without exactly one ':', line: %s", fileName, onePart, line));
            }
            int partId = Integer.parseInt(parts1[0]);
            if (partId != ind + 1) {
              throw new Exception(
                  String.format("Looks like there's a missing feature weight, field %d has id %d", ind + 1, partId));
            }
            res.set(ind, Double.parseDouble(parts1[1]));
            ind++;
          } catch (NumberFormatException e) {
            throw new Exception(
                String.format(
                    "The line in file '%s' has non-number '%s', line: %s", fileName, onePart, line));
          }
        }
        return res;
      }
      
    } finally {    
      inFile.close();
    }
    
    throw new Exception("No features found in '" + fileName + "'");
  }

  public static HashMap<String, DenseVector> initResultSet(CandidateEntry[] cands, int featureQty) {
    HashMap<String, DenseVector> res = new HashMap<String,DenseVector>();

    for (CandidateEntry e : cands) {
      res.put(e.mDocId, new DenseVector(featureQty));
    }
    
    return res;
  }
  
  public static DocEntryParsed getQueryEntry(String fieldName, 
                                             ForwardIndex fieldIndex, 
                                             DataEntryFields queryFields) {
    String query = queryFields.getString(fieldName);
    if (null == query) return null;
    query = query.trim();
    if (query.isEmpty()) return null;
    
    return
        fieldIndex.createDocEntryParsed(query.split("\\s+"),
                                  true  /* True means we generate word ID sequence:
                                   * in the case of queries, there's never a harm in doing so.
                                   * If word ID sequence is not used, it will be used only to compute the document length. */              
                                  );
  }
}
