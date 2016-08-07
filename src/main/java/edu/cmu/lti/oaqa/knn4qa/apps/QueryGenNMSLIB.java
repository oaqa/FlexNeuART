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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.*;
import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.xml.sax.SAXException;

import edu.cmu.lti.oaqa.annographix.util.*;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.NmslibQueryGenerator;

/**
 * An application to generate queries that can be processed by NMSLIB.
 * 
 * @author Leonid Boytsov
 *
 */
public class QueryGenNMSLIB {   
  private final static String NL = System.getProperty("line.separator");
  
  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "QueryGenNMSLIB", options );      
    System.exit(1);
  }
  
  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.QUERY_FILE_PARAM,    null, true, CommonParams.QUERY_FILE_DESC);
    options.addOption(CommonParams.MEMINDEX_PARAM,      null, true, CommonParams.MEMINDEX_DESC);
    options.addOption(CommonParams.KNN_QUERIES_PARAM,   null, true, CommonParams.KNN_QUERIES_DESC);
    options.addOption(CommonParams.NMSLIB_FIELDS_PARAM, null, true, CommonParams.NMSLIB_FIELDS_DESC);
    options.addOption(CommonParams.MAX_NUM_QUERY_PARAM, null, true, CommonParams.MAX_NUM_QUERY_DESC);
    options.addOption(CommonParams.SEL_PROB_PARAM,      null, true, CommonParams.SEL_PROB_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();     

    
    BufferedWriter knnQueries = null;    
    
    int maxNumQuery = Integer.MAX_VALUE;
    
    Float selProb = null;
    
    try {
      CommandLine cmd = parser.parse(options, args);
      String queryFile= null;
          
      if (cmd.hasOption(CommonParams.QUERY_FILE_PARAM)) {
        queryFile = cmd.getOptionValue(CommonParams.QUERY_FILE_PARAM);
      } else {
        Usage("Specify 'query file'", options);
      }
      
      String knnQueriesFile = cmd.getOptionValue(CommonParams.KNN_QUERIES_PARAM);
      
      if (null == knnQueriesFile)
        Usage("Specify '" + CommonParams.KNN_QUERIES_DESC+ "'", options);
      
      String tmpn = cmd.getOptionValue(CommonParams.MAX_NUM_QUERY_PARAM);
      if (tmpn != null) {
        try {
          maxNumQuery = Integer.parseInt(tmpn);
        } catch (NumberFormatException e) {
          Usage("Maximum number of queries isn't integer: '" + tmpn + "'", options);
        }
      }      
      
      String tmps = cmd.getOptionValue(CommonParams.NMSLIB_FIELDS_PARAM);
      if (null == tmps)
        Usage("Specify '" + CommonParams.NMSLIB_FIELDS_DESC + "'", options);
      String nmslibFieldList[] = tmps.split(",");
      
      knnQueries = new BufferedWriter(new FileWriter(knnQueriesFile));
      knnQueries.write("isQueryFile=1");
      knnQueries.newLine();
      knnQueries.newLine();
      
      String memIndexPref = cmd.getOptionValue(CommonParams.MEMINDEX_PARAM);
      
      if (null == memIndexPref) {
        Usage("Specify '" + CommonParams.MEMINDEX_DESC + "'", options);
      }
      
      String tmpf = cmd.getOptionValue(CommonParams.SEL_PROB_PARAM);
      
      if (tmpf != null) {
        try {
          selProb = Float.parseFloat(tmpf);
        } catch (NumberFormatException e) {
          Usage("A selection probability isn't a number in the range (0,1)'" + tmpf + "'", options);
        }
        if (selProb < Float.MIN_NORMAL || selProb + Float.MIN_NORMAL >= 1)
          Usage("A selection probability isn't a number in the range (0,1)'" + tmpf + "'", options);
      }
            
      BufferedReader  inpText = new BufferedReader(
          new InputStreamReader(CompressUtils.createInputStream(queryFile)));
      
      String docText = XmlHelper.readNextXMLIndexEntry(inpText);
      
      NmslibQueryGenerator queryGen = new NmslibQueryGenerator(nmslibFieldList, memIndexPref); 

      Random rnd = new Random();
      
      for (int docNum = 1; 
           docNum <= maxNumQuery && docText!= null; 
           ++docNum, docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
        if (selProb != null) {
          if (rnd.nextFloat() > selProb) continue;
        }
        
        Map<String, String>         docFields = null; 
          
        try {
          docFields = XmlHelper.parseXMLIndexEntry(docText);
          
          String queryObjStr = queryGen.getStrObjForKNNService(docFields);          

          knnQueries.append(queryObjStr);
          knnQueries.newLine();
        } catch (SAXException e) {
          System.err.println("Parsing error, offending DOC:" + NL + docText + " doc # " + docNum);
          throw new Exception("Parsing error.");
        }       
      }
            
     	knnQueries.close();
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
      if (null != knnQueries)
        try {
          knnQueries.close();
        } catch (IOException e1) {
          e1.printStackTrace();
        }
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      try {
        if (knnQueries != null) knnQueries.close();
      } catch (IOException e1) {
        e1.printStackTrace();
      }      
      System.exit(1);
    } 
    
    System.out.println("Terminated successfully!");
  }  
}
