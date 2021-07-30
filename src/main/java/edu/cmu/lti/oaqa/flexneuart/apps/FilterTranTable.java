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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.FrequentIndexWordFilterAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.VocabularyFilterAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaTranRec;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;


public class FilterTranTable {  
  static final Logger logger = LoggerFactory.getLogger(FilterTranTable.class);
  
  private final static boolean BINARY_OUTUPT = true;


  private static final String INPUT_PARAM = "i";
  private static final String INPUT_DESC = "A name of the input translation file (can have a .gz or .bz2 extension)";

  private static final String OUTPUT_PARAM = "o";
  private static final String OUTPUT_DESC  = "A name of the output file (can have a .gz extension to be compressed)";
   

  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(mAppName, mOptions);      
    System.exit(1);
  }
  static void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }
  
  static Options mOptions = new Options();
  static String  mAppName = "Filter translation table";
  
  public static void main(String[] args) {

    
    mOptions.addOption(INPUT_PARAM,                        null, true, INPUT_DESC);
    mOptions.addOption(OUTPUT_PARAM,                       null, true, OUTPUT_DESC);

    mOptions.addOption(CommonParams.FLT_FWD_INDEX_PARAM,   null, true, CommonParams.FLT_FWD_INDEX_DESC);
    mOptions.addOption(CommonParams.GIZA_ITER_QTY_PARAM,   null, true, CommonParams.GIZA_ITER_QTY_DESC);
    mOptions.addOption(CommonParams.MODEL1_ROOT_DIR_PARAM, null, true, CommonParams.MODEL1_ROOT_DIR_DESC);
    mOptions.addOption(CommonParams.MIN_PROB_PARAM,        null, true, CommonParams.MIN_PROB_DESC);
    mOptions.addOption(CommonParams.MAX_WORD_QTY_PARAM,    null, true, CommonParams.MAX_WORD_QTY_PARAM);  
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();

    try {
      CommandLine cmd = parser.parse(mOptions, args);
      
      String outputFile = null;
      
      outputFile = cmd.getOptionValue(OUTPUT_PARAM);
      if (null == outputFile) {
        showUsageSpecify(OUTPUT_PARAM);
      }
      
      String model1RootDir = cmd.getOptionValue(CommonParams.MODEL1_ROOT_DIR_PARAM);
      if (null == model1RootDir) {
        showUsageSpecify(CommonParams.MODEL1_ROOT_DIR_PARAM + "'");
      }
      
      String gizaIterQty = cmd.getOptionValue(CommonParams.GIZA_ITER_QTY_PARAM);
      
      if (null == gizaIterQty) {
        showUsageSpecify(CommonParams.GIZA_ITER_QTY_DESC);
      }
            

      float minProb = 0;
            
      String tmpf = cmd.getOptionValue(CommonParams.MIN_PROB_PARAM);
      
      if (tmpf != null) {
        minProb = Float.parseFloat(tmpf);
      }
      
      
      int maxWordQty = Integer.MAX_VALUE;
      
      String tmpi = cmd.getOptionValue(CommonParams.MAX_WORD_QTY_PARAM);
      
      if (null != tmpi) {
        maxWordQty = Integer.parseInt(tmpi);
      }
                 
      String fwdIndxName = cmd.getOptionValue(CommonParams.FLT_FWD_INDEX_PARAM);
      if (null == fwdIndxName) {
        showUsageSpecify(CommonParams.FLT_FWD_INDEX_DESC);
      }      

      logger.info("Filtering index: " + fwdIndxName + " max # of frequent words: " + maxWordQty + " min. probability:" + minProb);
      
      VocabularyFilterAndRecoder filter 
            = new FrequentIndexWordFilterAndRecoder(fwdIndxName, maxWordQty);
      
      String srcVocFile = CompressUtils.findFileVariant(model1RootDir + File.separator + "source.vcb");
      
      logger.info("Source vocabulary file: " + srcVocFile);
      
      GizaVocabularyReader srcVoc = 
          new GizaVocabularyReader(srcVocFile, filter);
      
      String dstVocFile = CompressUtils.findFileVariant(model1RootDir + File.separator + "target.vcb");
      
      logger.info("Target vocabulary file: " + dstVocFile);
      
      GizaVocabularyReader dstVoc = 
          new GizaVocabularyReader(CompressUtils.findFileVariant(dstVocFile), filter);            

      String inputFile = CompressUtils.findFileVariant(model1RootDir + File.separator + "output.t1." + gizaIterQty);
      
      BufferedReader finp = new BufferedReader(new InputStreamReader(
                                                CompressUtils.createInputStream(inputFile)));
      
      BufferedWriter fout = null;
      DataOutputStream foutBin = null;
      if (BINARY_OUTUPT) {
        foutBin = new DataOutputStream(
                        new BufferedOutputStream(CompressUtils.createOutputStream(
                                          GizaTranTableReaderAndRecoder.binaryFileName(outputFile))));
      } else {
        fout = new BufferedWriter(new OutputStreamWriter(
                                  CompressUtils.createOutputStream(outputFile)));
      }
      
      try {
        String  line;
        int     prevSrcId = -1;
        int     wordQty = 0;
        long    addedQty = 0;
        long    totalQty = 0;
        boolean isNotFiltered = false;
        
        for (totalQty = 0; (line = finp.readLine()) != null; ) {
          ++totalQty;
          // Skip empty lines
          line = line.trim(); if (line.isEmpty()) continue;
          
          GizaTranRec rec = new GizaTranRec(line);
          
          if (rec.mSrcId != prevSrcId) {
            ++wordQty;
          }
          if (totalQty % (10 * Const.PROGRESS_REPORT_QTY) == 0) {
            logger.info(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
                                              totalQty, wordQty, inputFile, addedQty));
          }
          
          // isNotFiltered should be set after procOneWord
          if (rec.mSrcId != prevSrcId) {
            if (rec.mSrcId == 0) isNotFiltered = true;
            else {
              String wordSrc = srcVoc.getWord(rec.mSrcId);       
              isNotFiltered = filter == null || (wordSrc != null && filter.checkWord(wordSrc));
            }
          }           

          prevSrcId = rec.mSrcId;

          if (rec.mProb >= minProb && isNotFiltered) {
            String wordDst = dstVoc.getWord(rec.mDstId);

            if (filter == null || (wordDst != null && filter.checkWord(wordDst))) {
              if (BINARY_OUTUPT) {
                foutBin.writeInt(rec.mSrcId );   
                foutBin.writeInt(rec.mDstId);
                foutBin.writeFloat(rec.mProb);
              } else {
                fout.write(String.format(rec.mSrcId + " " + rec.mDstId + " " + rec.mProb));
                fout.newLine();   
              }

              addedQty++;
            }
          }          
        }
        
        logger.info(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
            totalQty, wordQty, inputFile, addedQty));
        
      } finally {
        finp.close();
        if (fout != null) {
          fout.close();
        }
        if (foutBin != null) {
          foutBin.close();
        }
      }      
    } catch (ParseException e) {
      showUsage("Cannot parse arguments");
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }
  
   
}
