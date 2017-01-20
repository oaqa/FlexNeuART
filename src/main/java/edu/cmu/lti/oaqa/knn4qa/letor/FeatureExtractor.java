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

import no.uib.cipr.matrix.DenseVector;

public abstract class FeatureExtractor {
  public static final int TEXT_FIELD_ID        = 0;
  public static final int TEXT_UNLEMM_FIELD_ID = 1;
  public static final int BIGRAM_FIELD_ID      = 2;
    
  public static final float BM25_K1 = 1.2f;
  public static final float BM25_B = 0.75f;
  
  public static final String QFEAT_ONLY = "qfeat_only";
  public static final String TEXT_QFEAT = "text_qfeat";
  public static final String EPHYRA_SPACY = "ephyra_spacy";
  public static final String EPHYRA_DBPEDIA = "ephyra_dbpedia";
  public static final String EPHYRA_ALLENT = "ephyra_allent";
  public static final String LEXICAL_SPACY = "lexical_spacy";
  public static final String LEXICAL_DBPEDIA = "lexical_dbpedia";
  public static final String LEXICAL_ALLENT = "lexical_allent";
  
  public boolean isSomeTextFieldId(int fieldId) {
    return fieldId == TEXT_FIELD_ID || fieldId == TEXT_UNLEMM_FIELD_ID;
  }
  
  public abstract String getName();

  /*
   * Field names in mFieldsSOLR are kept mostly for historical reasons.
   * The first candidate provider that we used was SOLR (which you can still use if you want). 
   * 1) These field names are specified in the SOLR configuration file (see solr/yahoo_answ/conf/schema.xml).
   * 2) These field names are specified in UIMA descriptors (see src/main/resources/descriptors).
   * 
   */
  public final static String[] mFieldsSOLR = {"Text_bm25",
                                              "TextUnlemm_bm25",
                                              "BiGram_bm25",
                                              "Srl_bm25",   
                                              "SrlLab_bm25",
                                              "DepRel_bm25",
                                              "WNSS_bm25",
                                              "Text_alias1",
                                              QFEAT_ONLY,
                                              TEXT_QFEAT,
                                              EPHYRA_SPACY,
                                              EPHYRA_DBPEDIA,
                                              EPHYRA_ALLENT,
                                              LEXICAL_SPACY,
                                              LEXICAL_DBPEDIA,
                                              LEXICAL_ALLENT,
                                              }; 

  public final static String[] mFieldNames = { 
                                            "text",
                                            "text_unlemm",
                                            "bigram",
                                            "srl",
                                            "srl_lab",
                                            "dep",
                                            "wnss",
                                            "text_alias1",
                                            QFEAT_ONLY,
                                            TEXT_QFEAT,
                                            EPHYRA_SPACY,
                                            EPHYRA_DBPEDIA,
                                            EPHYRA_ALLENT,
                                            LEXICAL_SPACY,
                                            LEXICAL_DBPEDIA,
                                            LEXICAL_ALLENT,
                                                                                        };
  /*
   * If a field is an alias of a certain field, the ID of the field it mirrors is given here.
   * LIMITATION: the alias should always be initialized after the field it mirrors. In other
   * words, an alias index/id should be larger.
   */
  public final static int[] mAliasOfId = {
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        TEXT_FIELD_ID,
                                        -1, // QFEAT_ONLY,
                                        -1, // TEXT_QFEAT,
                                        -1, // EPHYRA_SPACY,
                                        -1, // EPHYRA_DBPEDIA,
                                        -1, // EPHYRA_ALLENT,
                                        -1, // LEXICAL_SPACY,
                                        -1, // LEXICAL_DBPEDIA,
                                        -1, // LEXICAL_ALLENT,
  };

  /*
   * OOV_PROB is taken from
   * 
   * Learning to Rank Answers to Non-Factoid Questions from Web Collections
   * by Mihai Surdeanu et al.
   * 
   */
  public static final double OOV_PROB = 1e-9;
  public static final float DEFAULT_PROB_SELF_TRAN = 0.5f;
    
  
  public static String indexFileName(String prefixDir, String fileName) {
    return prefixDir + "/" + fileName;
  }  
  
  /**
   * Obtains features for a set of documents, this function should be <b>thread-safe!</b>.
   * 
   * @param     arrDocIds    an array of document IDs
   * @param     queryData    several pieces of input data, one is typically a bag-of-words query. 

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
}
