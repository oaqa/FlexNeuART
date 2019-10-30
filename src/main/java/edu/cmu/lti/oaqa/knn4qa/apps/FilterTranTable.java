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

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.cli.*;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.FrequentIndexWordFilterAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.VocabularyFilterAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranRec;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;


public class FilterTranTable {
  
  private static final int REPORT_INTERVAL_QTY = 100000;
  
  private final static boolean BINARY_OUTUPT = true;


  private static final String INPUT_PARAM = "i";
  private static final String INPUT_DESC = "A name of the input translation file (can have a .gz or .bz2 extension)";

  private static final String OUTPUT_PARAM = "o";
  private static final String OUTPUT_DESC  = "A name of the output file (can have a .gz extension to be compressed)";
   

  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("FilterTranTable", options );      
    System.exit(1);
  }  

  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(INPUT_PARAM,                      null, true, INPUT_DESC);
    options.addOption(OUTPUT_PARAM,                     null, true, OUTPUT_DESC);
    options.addOption(CommonParams.FLT_FWD_INDEX_PARAM, null, true, CommonParams.FLT_FWD_INDEX_DESC);
    options.addOption(CommonParams.GIZA_ITER_QTY_PARAM, null, true, CommonParams.GIZA_ITER_QTY_PARAM);
    options.addOption(CommonParams.GIZA_ROOT_DIR_PARAM, null, true, CommonParams.GIZA_ROOT_DIR_PARAM);
    options.addOption(CommonParams.MIN_PROB_PARAM,      null, true, CommonParams.MIN_PROB_DESC);
    options.addOption(CommonParams.MAX_WORD_QTY_PARAM,  null, true, CommonParams.MAX_WORD_QTY_PARAM);  
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();

    try {
      CommandLine cmd = parser.parse(options, args);
      
      String outputFile = null;
      
      outputFile = cmd.getOptionValue(OUTPUT_PARAM);
      if (null == outputFile) {
        Usage("Specify 'A name of the output file'", options);
      }
      
      String gizaRootDir = cmd.getOptionValue(CommonParams.GIZA_ROOT_DIR_PARAM);
      if (null == gizaRootDir) {
        Usage("Specify '" + CommonParams.GIZA_ROOT_DIR_DESC + "'", options);
      }
      
      String gizaIterQty = cmd.getOptionValue(CommonParams.GIZA_ITER_QTY_PARAM);
      
      if (null == gizaIterQty) {
        Usage("Specify '" + CommonParams.GIZA_ITER_QTY_DESC + "'", options);
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
        Usage("Specify '" + CommonParams.FLT_FWD_INDEX_DESC + "'", options);
      }      

      System.out.println("Filtering index: " + fwdIndxName + " max # of frequent words: " + maxWordQty + " min. probability:" + minProb);
      
      VocabularyFilterAndRecoder filter 
            = new FrequentIndexWordFilterAndRecoder(fwdIndxName, maxWordQty);
      
      String srcVocFile = CompressUtils.findFileVariant(gizaRootDir + File.separator + "source.vcb");
      
      System.out.println("Source vocabulary file: " + srcVocFile);
      
      GizaVocabularyReader srcVoc = 
          new GizaVocabularyReader(srcVocFile, filter);
      
      String dstVocFile = CompressUtils.findFileVariant(gizaRootDir + File.separator + "target.vcb");
      
      System.out.println("Target vocabulary file: " + dstVocFile);
      
      GizaVocabularyReader dstVoc = 
          new GizaVocabularyReader(CompressUtils.findFileVariant(dstVocFile), filter);            

      String inputFile = CompressUtils.findFileVariant(gizaRootDir + File.separator + "output.t1." + gizaIterQty);
      
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
          if (totalQty % REPORT_INTERVAL_QTY == 0) {
            System.out.println(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
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
        
        System.out.println(String.format("Processed %d lines (%d source word entries) from '%s', added %d lines", 
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
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }
  
   
}
