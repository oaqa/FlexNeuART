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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.FrequentIndexWordFilterAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.VocabularyFilterAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.giza.*;
import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.ParamHelper;

public class FilterVocabulary {
  static final Logger logger = LoggerFactory.getLogger(FilterVocabulary.class);
  
  private static final String OUT_VOC_FILE_DESC = "a name of the output file (can have a .gz extension to be compressed)";
  private static final String OUT_VOC_FILE_PARAM = "o";
  
  private static final String IN_VOC_FILE_DESC = "a name of the input vocabulary file (can have a .gz or .bz2 extension)";
  private static final String IN_VOC_FILE_PARAM = "i";
   
      
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

  public static void main(String[] args) {
    String optKeys[]  = {IN_VOC_FILE_PARAM, OUT_VOC_FILE_PARAM, CommonParams.FLT_FWD_INDEX_PARAM, CommonParams.MAX_WORD_QTY_PARAM};
    String optDescs[] = {IN_VOC_FILE_DESC,  OUT_VOC_FILE_DESC,  CommonParams.FLT_FWD_INDEX_DESC,  CommonParams.MAX_WORD_QTY_DESC};
    boolean hasArg[]  = {true,              true,               true,                             true};
  
    ParamHelper mParamHelper = null;
    
    try {

      mParamHelper = new ParamHelper(args, optKeys, optDescs, hasArg);    
   
      CommandLine cmd = mParamHelper.getCommandLine();
      
      String outputFile = cmd.getOptionValue(OUT_VOC_FILE_PARAM);      
      if (null == outputFile) {
        UsageSpecify(OUT_VOC_FILE_DESC, mParamHelper.getOptions());
      }
      
      String inputFile = cmd.getOptionValue(IN_VOC_FILE_PARAM);
      if (null == inputFile) {
        UsageSpecify(IN_VOC_FILE_DESC, mParamHelper.getOptions());
      }
      
      int maxWordQty = Integer.MAX_VALUE;
      
      String tmpi = cmd.getOptionValue(CommonParams.MAX_WORD_QTY_PARAM);
      
      if (null != tmpi) {
        maxWordQty = Integer.parseInt(tmpi);
      }
                  
      String fwdIndxName = cmd.getOptionValue(CommonParams.FLT_FWD_INDEX_PARAM);
      if (null == fwdIndxName) {
        UsageSpecify(CommonParams.FLT_FWD_INDEX_DESC, mParamHelper.getOptions());
      }
      
      VocabularyFilterAndRecoder filter = new FrequentIndexWordFilterAndRecoder(fwdIndxName, maxWordQty);

      BufferedReader finp = new BufferedReader(new InputStreamReader(
                                                CompressUtils.createInputStream(inputFile)));      
      BufferedWriter fout = new BufferedWriter(new OutputStreamWriter(
                                                CompressUtils.createOutputStream(outputFile)));      
      try {
        
        
        String  line;

        int     wordQty = 0;
        long    addedQty = 0;
        long    totalQty = 0;
        
        for (totalQty = 0; (line = finp.readLine()) != null; ) {
          ++totalQty;
          // Skip empty lines
          line = line.trim(); if (line.isEmpty()) continue;
          
          GizaVocRec rec = new GizaVocRec(line);
          
          if (filter.checkWord(rec.mWord)) {
            rec.save(fout);
            addedQty++;
          }
        
          if (totalQty % Const.PROGRESS_REPORT_QTY == 0)
            logger.info(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
                                       totalQty, wordQty, inputFile, addedQty));
        }
        logger.info(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
            totalQty, wordQty, inputFile, addedQty));
        
      } finally {
        finp.close();
        fout.close();
      }      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", mParamHelper != null ? mParamHelper.getOptions() : null);
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }
   
}
