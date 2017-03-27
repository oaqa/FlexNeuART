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

import org.apache.commons.cli.*;

import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;

import java.io.*;
import java.util.*;

/**
 * An simple application that reads text files produced by an annotation
 * pipeline and saves the value of a given field to an output file.
 * 
 * @author Leonid Boytsov
 *
 */

public class ExtractFieldValue {
  private static final String FIELD_NAME_PARAM = "field";
  private static final String FIELD_NAME_DESC  = "a field name to use (e.g., text)";
  private static final String INPUT_FILE_PARAM = "input";
  private static final String INPUT_FILE_DESC  = "input file";
  private static final String OUTPUT_FILE_PARAM= "output";
  private static final String OUTPUT_FILE_DESC = "output_file";

  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "LuceneIndexer", opt);      
    System.exit(1);
  }  
  
  public static void main(String [] args) {
    Options options = new Options();
    
    options.addOption(INPUT_FILE_PARAM,  null, true, INPUT_FILE_DESC);   
    options.addOption(OUTPUT_FILE_PARAM, null, true, OUTPUT_FILE_DESC);
    options.addOption(FIELD_NAME_PARAM,  null, true, FIELD_NAME_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
    
      String inputFileName = cmd.getOptionValue(INPUT_FILE_PARAM);      
      if (null == inputFileName) Usage("Specify: " + INPUT_FILE_PARAM, options);
      
      String outputFileName = cmd.getOptionValue(OUTPUT_FILE_PARAM);
      if (null == outputFileName) Usage("Specify: " + OUTPUT_FILE_PARAM, options);
      
      int fieldId = -1;
      
      String tmp = cmd.getOptionValue(FIELD_NAME_PARAM);
      
      if (tmp != null) {
        for (int i = 0; i < FeatureExtractor.mFieldNames.length; ++i) {
          if (0 == tmp.compareToIgnoreCase(FeatureExtractor.mFieldNames[i]) ||
              0 == tmp.compareToIgnoreCase(FeatureExtractor.mFieldsSOLR[i])) {
            fieldId = i;
            break;
          }          
        }        
      } else Usage("Specify: " + FIELD_NAME_PARAM, options);
      
      if (fieldId == -1) {
        Usage("Unknown field: " + tmp, options);
      }

      String fieldNameSOLR = FeatureExtractor.mFieldsSOLR[fieldId];
      
      BufferedReader inpText = new BufferedReader(new InputStreamReader(
          CompressUtils.createInputStream(inputFileName)));
      String docText = XmlHelper.readNextXMLIndexEntry(inpText);
      
      System.out.println("SOLR field: " + fieldNameSOLR);
      
      BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(outputFileName)));
      
      for (int docNum = 0; docText != null ; docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
        Map<String, String> docFields = null;

        try {
          docFields = XmlHelper.parseXMLIndexEntry(docText);
        } catch (Exception e) {
          System.err.println(String.format("Parsing error, offending DOC #%d:\n%s", docNum, docText));
          System.exit(1);
        }

        String val = docFields.get(fieldNameSOLR);
        val = val == null ? "" : val;
        
        outFile.write(val); outFile.newLine();
        
        ++docNum;
      }       
      
      outFile.close();
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
  }
}
