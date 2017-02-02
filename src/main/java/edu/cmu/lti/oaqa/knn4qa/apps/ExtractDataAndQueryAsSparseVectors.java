/*
 *  Copyright 2016 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntryExt;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.simil.TrulySparseVector;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.ParamHelper;

public class ExtractDataAndQueryAsSparseVectors {

  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    if (options != null) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("FilterTranTable", options );
    }
    System.exit(1);
  }  
  
  static void UsageSpecify(String param, Options options) {
    Usage("Specify '" + param + "'", options);
  }  
  
  public static String MAX_NUM_DATA_PARAM = "max_num_data";
  public static String MAX_NUM_DATA_DESC  = "maximum number of data points to extract";
  
  public static String IN_QUERIES_PARAM   = "in_queries";
  public static String IN_QUERIES_DESC    = "the input query file to be processed (in XML format)";
  
  public static String OUT_QUERIES_PARAM  = "out_queries";
  public static String OUT_QUERIES_DESC   = "the output sparse vector file with queries";
  
  public static String OUT_DATA_PARAM     = "out_data";
  public static String OUT_DATA_DESC      = "the output sparse vector file with data points";
  
  public static String TEXT_FIELD_PARAM   = "field";
  public static String TEXT_FIELD_DESC    = "the field name to process";
  
  public static String TEST_QTY_PARAM     = "test_qty";
  public static String TEST_QTY_DESC      = "the number of documents and queries for cross-validation of the extraction accuracy";
   
  
  public static void main(String[] args) {
    String optKeys[] = {
        CommonParams.MAX_NUM_QUERY_PARAM,
        MAX_NUM_DATA_PARAM,
        CommonParams.MEMINDEX_PARAM,
        IN_QUERIES_PARAM,
        OUT_QUERIES_PARAM,
        OUT_DATA_PARAM,
        TEXT_FIELD_PARAM,
        TEST_QTY_PARAM,
    };
    String optDescs[] = {
        CommonParams.MAX_NUM_QUERY_DESC,
        MAX_NUM_DATA_DESC,
        CommonParams.MEMINDEX_DESC,
        IN_QUERIES_DESC,
        OUT_QUERIES_DESC,
        OUT_DATA_DESC,  
        TEXT_FIELD_DESC,
        TEST_QTY_DESC
    };
    boolean hasArg[] = {
        true,
        true,
        true,
        true,
        true,
        true,        
        true,
        true
    };
    
    ParamHelper prmHlp = null;
    
    try {

      prmHlp = new ParamHelper(args, optKeys, optDescs, hasArg);    
   
      CommandLine cmd = prmHlp.getCommandLine();
      Options     opt = prmHlp.getOptions();
      
      int maxNumQuery = Integer.MAX_VALUE;

      String tmpn = cmd.getOptionValue(CommonParams.MAX_NUM_QUERY_PARAM);
      if (tmpn != null) {
        try {
          maxNumQuery = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          UsageSpecify(CommonParams.MAX_NUM_QUERY_PARAM, opt);
        }
      }
      
      int maxNumData = Integer.MAX_VALUE;
      tmpn = cmd.getOptionValue(MAX_NUM_DATA_PARAM);
      if (tmpn != null) {
        try {
          maxNumData = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          UsageSpecify(MAX_NUM_DATA_PARAM, opt);
        }
      }
      String memIndexPref = cmd.getOptionValue(CommonParams.MEMINDEX_PARAM);
      if (null == memIndexPref) {
        UsageSpecify(CommonParams.MEMINDEX_PARAM, opt);
      }
      String textField = cmd.getOptionValue(TEXT_FIELD_PARAM);
      if (null == textField) {
        UsageSpecify(TEXT_FIELD_PARAM, opt);
      }
      
      textField = textField.toLowerCase();
      int fieldId = -1;
      for (int i = 0; i < FeatureExtractor.mFieldNames.length; ++i) 
      if (FeatureExtractor.mFieldNames[i].compareToIgnoreCase(textField)==0) {
        fieldId = i;
        break;
      }
      if (-1 == fieldId) {
        Usage("Wrong field index, should be one of the following: " + String.join(",", FeatureExtractor.mFieldNames), opt);
      }
      
      InMemForwardIndex indx =
          new InMemForwardIndex(FeatureExtractor.indexFileName(memIndexPref, FeatureExtractor.mFieldNames[fieldId]));
      
      BM25SimilarityLucene bm25simil = 
          new BM25SimilarityLucene(FeatureExtractor.BM25_K1, FeatureExtractor.BM25_B, indx);
      
      String inQueryFile = cmd.getOptionValue(IN_QUERIES_PARAM);
      String outQueryFile = cmd.getOptionValue(OUT_QUERIES_PARAM);
      if ((inQueryFile == null) != (outQueryFile == null)) {
        Usage("You should either specify both " + IN_QUERIES_PARAM + " and " + OUT_QUERIES_PARAM + " or none of them", opt);
      }
      String outDataFile = cmd.getOptionValue(OUT_DATA_PARAM);
      
      tmpn = cmd.getOptionValue(TEST_QTY_PARAM);
      int testQty = 0;
      if (tmpn != null) {
        try {
          testQty = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          UsageSpecify(TEST_QTY_PARAM, opt);
        }
      }

      ArrayList<DocEntry>           testDocEntries = new ArrayList<DocEntry>();
      ArrayList<DocEntry>           testQueryEntries = new ArrayList<DocEntry>();
      ArrayList<TrulySparseVector>  testDocVectors = new ArrayList<TrulySparseVector>();
      ArrayList<TrulySparseVector>  testQueryVectors = new ArrayList<TrulySparseVector>();
      
      if (outDataFile != null) {
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outDataFile)));
        
        ArrayList<DocEntryExt> docEntries = indx.getDocEntries();
        
        for (int id = 0; id < Math.min(maxNumData, docEntries.size()); ++id) {
          DocEntry e = docEntries.get(id).mDocEntry;
          TrulySparseVector v = bm25simil.getDocSparseVector(e, false);
          if (id < testQty) {
            testDocEntries.add(e);
            testDocVectors.add(v);
          }
          outputVector(out, v);
        }

        out.close();        
      }

      Splitter splitOnSpace = Splitter.on(' ').trimResults().omitEmptyStrings();

      
      if (outQueryFile != null) {
        BufferedReader inpText = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inQueryFile)));
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outQueryFile)));      

        
        String queryText = XmlHelper.readNextXMLIndexEntry(inpText);        

        for (int queryQty = 0; queryText!= null && queryQty < maxNumQuery; 
            queryText = XmlHelper.readNextXMLIndexEntry(inpText), queryQty++) {          
          Map<String, String>    queryFields = null;
          // 1. Parse a query

          try {
            queryFields = XmlHelper.parseXMLIndexEntry(queryText);
          } catch (Exception e) {
            System.err.println("Parsing error, offending QUERY:\n" + queryText);
            throw new Exception("Parsing error.");
          }
          
          String fieldText = queryFields.get(FeatureExtractor.mFieldsSOLR[fieldId]);
          
          if (fieldText == null) {
            fieldText = "";
          }
          
          ArrayList<String> tmpa = new ArrayList<String>();
          for (String s: splitOnSpace.split(fieldText)) tmpa.add(s);
          
          DocEntry e = indx.createDocEntry(tmpa.toArray(new String[tmpa.size()]), false /* no pos. info needed */);
          
          TrulySparseVector v = bm25simil.getDocSparseVector(e, true);
          if (queryQty < testQty) {
            testQueryEntries.add(e);
            testQueryVectors.add(v);
          }
          outputVector(out, v);            
        }
                
        out.close();
      }

      int testedQty = 0, diffQty = 0;
      // Now let's do some testing
      for (int iq = 0; iq < testQueryEntries.size(); ++iq) {
        DocEntry queryEntry = testQueryEntries.get(iq);
        TrulySparseVector queryVector = testQueryVectors.get(iq);
        for (int id = 0; id < testDocEntries.size(); ++id) {
          DocEntry docEntry = testDocEntries.get(id);
          TrulySparseVector docVector = testDocVectors.get(id);
          float val1 = bm25simil.compute(queryEntry, docEntry);
          float val2 = TrulySparseVector.scalarProduct(queryVector, docVector);
          ++testedQty;
          if (Math.abs(val1 - val2) > 1e5) {
            System.err.println(String.format("Potential mismatch BM25=%f <-> scalar product=%f", val1, val2));
            ++diffQty;
          }
        }
      }
      if (testedQty > 0) System.out.println(String.format("Tested %d Mismatched %d", testedQty, diffQty));
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments: " + e, prmHlp != null ? prmHlp.getOptions() : null);
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }        
  }

  private static void outputVector(BufferedWriter out, TrulySparseVector v) throws IOException {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < v.mIDs.length; ++i) {
      if (i>0) sb.append(' ');
      sb.append(v.mIDs[i]);
      sb.append(':');
      sb.append(v.mVals[i]);      
    }    
    String res = sb.toString().trim();
    if (!res.isEmpty()) {
      out.write(res);
      out.newLine();
    }
  }
  
}
