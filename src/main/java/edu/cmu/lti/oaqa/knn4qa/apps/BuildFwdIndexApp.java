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

import java.io.File;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.fwdindx.ForwardIndex.ForwardIndexType;

public class BuildFwdIndexApp {
  
  public final static String STORE_WORD_ID_SEQ_PARAM = "store_word_id_seq";
  public final static String STORE_WORD_ID_SEQ_DESC  = "Store positional info (a sequence of word IDs) in addition to word frequencies";  
  
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "BuildInMemFwdIndexApp", opt);     
    System.exit(1);
  }
  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.INPUT_DATA_DIR_PARAM,         null, true, CommonParams.INPUT_DATA_DIR_DESC);
    options.addOption(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM,           null, true, CommonParams.INPDATA_SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,            null, true, CommonParams.MAX_NUM_REC_DESC); 
    options.addOption(CommonParams.DATA_FILE_PARAM,              null, true, CommonParams.DATA_FILE_DESC);   
    options.addOption(CommonParams.OUT_INDEX_PARAM,              null, true, CommonParams.OUT_INDEX_DESC);
    options.addOption(CommonParams.FIELD_NAME_PARAM,             null, true, CommonParams.FIELD_NAME_DESC);
    options.addOption(CommonParams.FOWARD_INDEX_TYPE_PARAM,      null, true, CommonParams.FOWARD_INDEX_TYPE_DESC);
    options.addOption(STORE_WORD_ID_SEQ_PARAM,                   null, false, STORE_WORD_ID_SEQ_DESC);


    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String inputDataDir = null;
      
      inputDataDir = cmd.getOptionValue(CommonParams.INPUT_DATA_DIR_PARAM);
      
      if (null == inputDataDir) Usage("Specify: " + CommonParams.INPUT_DATA_DIR_PARAM, options);
      
      String outPrefix = cmd.getOptionValue(CommonParams.OUT_INDEX_PARAM);
      
      if (null == outPrefix) Usage("Specify: " + CommonParams.OUT_INDEX_PARAM, options);
      
      String subDirTypeList = cmd.getOptionValue(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM);
      
      if (null == subDirTypeList ||
          subDirTypeList.isEmpty()) Usage("Specify: " + CommonParams.INPDATA_SUB_DIR_TYPE_PARAM, options);
      
      String dataFileName = cmd.getOptionValue(CommonParams.DATA_FILE_PARAM);

      if (null == dataFileName) {
        Usage("Specify: " + CommonParams.DATA_FILE_PARAM, options);
      }
      
      int maxNumRec = Integer.MAX_VALUE;
      
      String tmp = cmd.getOptionValue(CommonParams.MAX_NUM_REC_PARAM);
      
      if (tmp != null) {
        try {
          maxNumRec = Integer.parseInt(tmp);
          if (maxNumRec <= 0) {
            Usage("The maximum number of records should be a positive integer", options);
          }
        } catch (NumberFormatException e) {
          Usage("The maximum number of records should be a positive integer", options);
        }
      }
      
      String fieldName = cmd.getOptionValue(CommonParams.FIELD_NAME_PARAM);
      if (fieldName == null) {
        Usage("Specify: '" + CommonParams.FIELD_NAME_PARAM, options);
      }
      
      String [] subDirs = subDirTypeList.split(",");

      System.out.println("Processing field: '" + fieldName + "'");
        
      String [] fileNames = new String[subDirs.length];
      for (int i = 0; i < fileNames.length; ++i)
        fileNames[i] = inputDataDir + File.separator + subDirs[i] + File.separator + dataFileName;
      
      boolean bStoreWordIdSeq = cmd.hasOption(STORE_WORD_ID_SEQ_PARAM);
      
      String fwdIndexType = cmd.getOptionValue(CommonParams.FOWARD_INDEX_TYPE_PARAM);
      
      if (fwdIndexType == null) {
        Usage("Specify: " + CommonParams.FOWARD_INDEX_TYPE_PARAM, options);
      }
      
      ForwardIndexType iType = ForwardIndex.getIndexType(fwdIndexType);
      
      if (iType == ForwardIndexType.unknown) {
        Usage("Wrong value for " + CommonParams.FOWARD_INDEX_TYPE_PARAM, options);
      }
      
      System.out.println("Forward index type: " + fwdIndexType);
      System.out.println("Storing word id sequence?: " + bStoreWordIdSeq);
        
      ForwardIndex indx = ForwardIndex.createWriteInstance(outPrefix + File.separator + fieldName, iType);
      
      indx.createIndex(fieldName, fileNames, bStoreWordIdSeq, maxNumRec);
      indx.saveIndex();
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }

}
