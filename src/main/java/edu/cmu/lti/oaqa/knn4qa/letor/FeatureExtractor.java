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
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.VectorWrapper;
import no.uib.cipr.matrix.DenseVector;

public abstract class FeatureExtractor {  
  public abstract String getName();
  
  /**
   * Obtains features for a set of documents, this function should be <b>thread-safe!</b>.
   * 
   * @param     arrDocIds    an array of document IDs
   * @param     queryData    a multifield representation of the query (map keys are field names). 

   * @return a map docId -> sparse feature vector
   */
  public abstract Map<String,DenseVector> getFeatures(ArrayList<String>    arrDocIds, 
                                                      Map<String, String>  queryData) throws Exception;
  
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


  public void normZeroOne(Map<String, DenseVector> docFeats) {
    int featureQty = this.getFeatureQty();
    
    double minVals[] = new double[featureQty];
    double maxVals[] = new double[featureQty];

    if (!docFeats.isEmpty()) {
      for (int i = 0; i < featureQty; ++i) {
        minVals[i] = Double.POSITIVE_INFINITY;
        maxVals[i] = Double.NEGATIVE_INFINITY;
      }
      // Let's 0-1 normalize
      for (Map.Entry<String, DenseVector> e : docFeats.entrySet()) {
        for (int i = 0; i < featureQty; ++i) {
          double val = e.getValue().get(i);
          minVals[i] = Math.min(val, minVals[i]);
          maxVals[i] = Math.max(val, maxVals[i]);
        }
      }

      for (int i = 0; i < featureQty; ++i) { 
        double diff = maxVals[i] - minVals[i];
        if (diff > Float.MIN_NORMAL) {
          for (Map.Entry<String, DenseVector> e : docFeats.entrySet()) {
            double val = e.getValue().get(i);
            e.getValue().set(i, (val - minVals[i]) / diff);
          }
        } else {
          for (Map.Entry<String, DenseVector> e : docFeats.entrySet())
            e.getValue().set(i, 0);
        }
      }
    }
  }

  public static HashMap<String, DenseVector> initResultSet(ArrayList<String> arrDocIds, int featureQty) {
    HashMap<String, DenseVector> res = new HashMap<String,DenseVector>();

    for (String docId : arrDocIds) {
      res.put(docId, new DenseVector(featureQty));
    }
    
    return res;
  }
  
  public static DocEntry getQueryEntry(String fieldName, ForwardIndex fieldIndex, Map<String, String> queryData) {
    String query = queryData.get(fieldName);
    if (null == query) return null;
    query = query.trim();
    if (query.isEmpty()) return null;
    
    return
        fieldIndex.createDocEntry(query.split("\\s+"),
                                  true  /* True means we generate word ID sequence:
                                   * in the case of queries, there's never a harm in doing so.
                                   * If word ID sequence is not used, it will be used only to compute the document length. */              
                                  );
  }
}
