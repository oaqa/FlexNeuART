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
package edu.cmu.lti.oaqa.knn4qa.apps;

import org.apache.commons.cli.*;

import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;

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

      String fieldName = cmd.getOptionValue(FIELD_NAME_PARAM);

      
      System.out.println("Indexing field: " + fieldName);
      
      BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(outputFileName)));
      Map<String, String> docFields = null;
      
      try (DataEntryReader inp = new DataEntryReader(inputFileName)) {
        for (int docNum = 0; ((docFields = inp.readNext()) != null) ; ++docNum) {

          String val = docFields.get(fieldName);
          val = val == null ? "" : val;
          
          outFile.write(val); outFile.newLine();
      
          
        }
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
