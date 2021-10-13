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

import java.io.File;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex.ForwardIndexStoreType;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex.ForwardIndexFieldType;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex.ForwardIndexType;

public class BuildFwdIndexApp {  
  
  private static final Logger logger = LoggerFactory.getLogger(BuildFwdIndexApp.class);
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "BuildInMemFwdIndexApp", opt);     
    System.exit(1);
  }
  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.INPUT_DATA_DIR_PARAM,          null, true, CommonParams.INPUT_DATA_DIR_DESC);
    options.addOption(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM,    null, true, CommonParams.INPDATA_SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,             null, true, CommonParams.MAX_NUM_REC_DESC); 
    options.addOption(CommonParams.EXPECTED_DOC_QTY_PARAM,        null, true, CommonParams.EXPECTED_DOC_QTY_DESC); 
    options.addOption(CommonParams.DATA_FILE_PARAM,               null, true, CommonParams.DATA_FILE_DESC);   
    options.addOption(CommonParams.OUT_INDEX_PARAM,               null, true, CommonParams.OUT_INDEX_DESC);
    options.addOption(CommonParams.FIELD_NAME_PARAM,              null, true, CommonParams.FIELD_NAME_DESC);
    options.addOption(CommonParams.FOWARD_INDEX_TYPE_PARAM,       null, true, CommonParams.FOWARD_INDEX_TYPE_DESC);
    options.addOption(CommonParams.FOWARD_INDEX_STORE_TYPE_PARAM, null, true, CommonParams.FOWARD_INDEX_STORE_TYPE_DESC);
    options.addOption(CommonParams.FOWARD_INDEX_FIELD_TYPE_PARAM, null, true, CommonParams.FOWARD_INDEX_FIELD_TYPE_DESC);

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
            Usage(CommonParams.MAX_NUM_REC_PARAM + " should be a positive integer", options);
          }
        } catch (NumberFormatException e) {
          Usage(CommonParams.MAX_NUM_REC_PARAM + " should be a positive integer", options);
        }
      }
      
      // There seems to be no harm in "over-specify" the number of expected entries as opposed to under-specifying,
      // which slows indexing down quite a bit
      int expectedQty = (int)1e9;
      
      tmp = cmd.getOptionValue(CommonParams.EXPECTED_DOC_QTY_PARAM);
      if (tmp != null) {
        try {
          expectedQty = (int)Float.parseFloat(tmp);
          if (expectedQty <= 0) {
            Usage(CommonParams.EXPECTED_DOC_QTY_PARAM + " should be a positive integer < " + Integer.MAX_VALUE, options);
          }
        } catch (NumberFormatException e) {
          Usage(CommonParams.EXPECTED_DOC_QTY_PARAM + " should be a positive integer", options);
        }
      }
      
            
      String fieldName = cmd.getOptionValue(CommonParams.FIELD_NAME_PARAM);
      if (fieldName == null) {
        Usage("Specify: '" + CommonParams.FIELD_NAME_PARAM, options);
      }
      
      String [] subDirs = subDirTypeList.split(",");

      logger.info("Processing field: '" + fieldName + "'");
        
      String [] fileNames = new String[subDirs.length];
      for (int i = 0; i < fileNames.length; ++i)
        fileNames[i] = inputDataDir + File.separator + subDirs[i] + File.separator + dataFileName;
      
      
      ForwardIndexType iIndexType = ForwardIndexType.dataDict;
      String fwdIndexType = cmd.getOptionValue(CommonParams.FOWARD_INDEX_TYPE_PARAM);
      
      if (fwdIndexType != null) {
        iIndexType = ForwardIndex.getIndexType(fwdIndexType);
        if (iIndexType == ForwardIndexType.unknown) {
          Usage("Wrong value '" + fwdIndexType + "' for " + CommonParams.FOWARD_INDEX_TYPE_PARAM, options);
        }
      }
      
      logger.info("Forward index type: " + iIndexType);
      
      String fwdIndexStoreType = cmd.getOptionValue(CommonParams.FOWARD_INDEX_STORE_TYPE_PARAM);
      ForwardIndexStoreType iStoreType = ForwardIndexStoreType.mapdb;
      
      if (fwdIndexStoreType != null) {
          
        iStoreType = ForwardIndex.getStoreType(fwdIndexStoreType);
      
        if (iStoreType == ForwardIndexStoreType.unknown) {
          Usage("Wrong value '" + fwdIndexStoreType + "' for " + CommonParams.FOWARD_INDEX_STORE_TYPE_PARAM, options);
        }
      }
      
      logger.info("Forward index storage type: " + iStoreType);
      
      String fwdIndexFieldType = cmd.getOptionValue(CommonParams.FOWARD_INDEX_FIELD_TYPE_PARAM);
      
      if (fwdIndexFieldType == null) {
        Usage("Specify: " + CommonParams.FOWARD_INDEX_FIELD_TYPE_PARAM, options);
      }                  

      ForwardIndexFieldType iFieldType = ForwardIndex.getIndexFieldType(fwdIndexFieldType);
      
      if (iFieldType == ForwardIndexFieldType.unknown) {
      	Usage("Wrong value '" + fwdIndexFieldType + "' for " + CommonParams.FOWARD_INDEX_FIELD_TYPE_PARAM, options);
      }
      
      logger.info("Forward index field type: " + iFieldType);
        
      ForwardIndex indx = ForwardIndex.createWriteInstance(outPrefix + File.separator + fieldName, 
                                                          iIndexType, iStoreType, iFieldType);
      
      indx.createIndex(fieldName, fileNames, maxNumRec, expectedQty);
      indx.saveIndex();
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      logger.error("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }

}
