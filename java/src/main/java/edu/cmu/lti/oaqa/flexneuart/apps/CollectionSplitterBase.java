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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;

public class CollectionSplitterBase {
  static final Logger logger = LoggerFactory.getLogger(CollectionSplitterBase.class);
  
  protected static final String PART_NAMES_DESC = "Comma separated part names, e.g., dev,test,train";
  protected static final String PART_NAMES_PARAM = "n";
  
  protected static final String PROB_VALS_PROB_DESC = "Comma separated probabilities e.g., 0.1,0.2,0.7.";
  protected static final String PROB_VALS_PARAM = "p";
  
  protected static final String OUTPUT_FILE_DESC = "Output file prefix";
  protected static final String OUTPUT_PREFIX_PARAM = "o";
  
  protected static final String INPUT_FILE_DESC = "Input file";
  protected static final String INPUT_FILE_PARAM = "i";
  
  protected static String            mInputFileName;
  protected static BufferedWriter [] mOutFiles;
  
  protected  static Options     mOptions;  
  protected  static String      mAppName;
  protected  static String []   mPartNames = null;
  
  protected  static ArrayList<Double> mProbs = new ArrayList<Double>();    

  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(mAppName, mOptions);      
    System.exit(1);
  }
  static void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }  

  static void parseOptions(String args[]) throws Exception {
    mOptions = new Options();
    
    mOptions.addOption(INPUT_FILE_PARAM,    null, true, INPUT_FILE_DESC);
    mOptions.addOption(OUTPUT_PREFIX_PARAM, null, true, OUTPUT_FILE_DESC);
    mOptions.addOption(PROB_VALS_PARAM,     null, true, PROB_VALS_PROB_DESC);
    mOptions.addOption(PART_NAMES_PARAM,    null, true, PART_NAMES_DESC);

    CommandLineParser parser = new org.apache.commons.cli.GnuParser(); 

    CommandLine cmd = parser.parse(mOptions, args);
    
    if (cmd.hasOption(INPUT_FILE_PARAM)) {
      mInputFileName = cmd.getOptionValue(INPUT_FILE_PARAM);
    } else {
      showUsageSpecify(INPUT_FILE_PARAM); 
    }

    if (cmd.hasOption(PROB_VALS_PARAM)) {
      String parts[] = cmd.getOptionValue(PROB_VALS_PARAM).split(",");

      try {
        double sum = 0;
        for (String s: parts) {
          double p = Double.parseDouble(s);
          if (p <= 0 || p > 1) showUsage("All probabilities must be in the range (0,1)");
          sum += p;
          mProbs.add(p);
        }
          
        if (Math.abs(sum - 1.0) > Float.MIN_NORMAL) {
          showUsage("The sum of probabilities should be equal to 1, but it's: " + sum);
        }
      } catch (NumberFormatException e ) {
        showUsage("Can't convert some of the probabilities to a floating-point number.");
      }
    } else {
      showUsageSpecify(PROB_VALS_PARAM);
    }
    
    if (cmd.hasOption(PART_NAMES_PARAM)) {
      mPartNames = cmd.getOptionValue(PART_NAMES_PARAM).split(",");
      
      if (mPartNames.length != mProbs.size())
        showUsage("The number of probabilities is not equal to the number of parts!");
    } else {
      showUsage("Specify part names");
    }
    
    mOutFiles = new BufferedWriter[mPartNames.length];  
    
    if (cmd.hasOption(OUTPUT_PREFIX_PARAM)) {
      String outPrefix = cmd.getOptionValue(OUTPUT_PREFIX_PARAM);

      for (int partId = 0; partId < mPartNames.length; ++partId) {
        mOutFiles[partId] =  new BufferedWriter(
                                  new OutputStreamWriter(
                                       CompressUtils.createOutputStream(outPrefix+"_" + mPartNames[partId]+ ".gz")));
      }
    } else {
      showUsage("Specify Output file prefix");      
    }
    
    logger.info("Using probabilities:");
    for (int partId = 0; partId < mPartNames.length; ++partId) 
      logger.info(mPartNames[partId] + " : " + mProbs.get(partId));
    logger.info("=================================================");  
    
  }
  protected static void closeFiles() throws IOException {
    for (BufferedWriter f : mOutFiles) f.close();
  }

}
