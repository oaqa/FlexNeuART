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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.InMemIndexFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;

public class BuildInMemFwdIndexApp {
  
  public final static String FIELD_NAME_PARAM = "field";
  public final static String FIELD_NAME_DESC  = "a field to be indexed (use a regular name not the one from the XML index-file)";  
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "BuildInMemFwdIndexApp", opt);     
    System.exit(1);
  }
  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.ROOT_DIR_PARAM,      null, true, CommonParams.ROOT_DIR_DESC);
    options.addOption(CommonParams.SUB_DIR_TYPE_PARAM,  null, true, CommonParams.SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,   null, true, CommonParams.MAX_NUM_REC_DESC);
    options.addOption(CommonParams.SOLR_FILE_NAME_PARAM,null, true, CommonParams.SOLR_FILE_NAME_DESC);    
    options.addOption(CommonParams.OUT_INDEX_PARAM,     null, true, CommonParams.OUT_MINDEX_DESC);
    options.addOption(FIELD_NAME_PARAM,                 null, true, FIELD_NAME_DESC);

    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String rootDir = null;
      
      rootDir = cmd.getOptionValue(CommonParams.ROOT_DIR_PARAM);
      
      if (null == rootDir) Usage("Specify: " + CommonParams.ROOT_DIR_DESC, options);
      
      String outPrefix = cmd.getOptionValue(CommonParams.OUT_INDEX_PARAM);
      
      if (null == outPrefix) Usage("Specify: " + CommonParams.OUT_MINDEX_DESC, options);
      
      String subDirTypeList = cmd.getOptionValue(CommonParams.SUB_DIR_TYPE_PARAM);
      
      if (null == subDirTypeList ||
          subDirTypeList.isEmpty()) Usage("Specify: " + CommonParams.SUB_DIR_TYPE_DESC, options);
      
      String solrFileName = cmd.getOptionValue(CommonParams.SOLR_FILE_NAME_PARAM);
      if (null == solrFileName) Usage("Specify: " + CommonParams.SOLR_FILE_NAME_DESC, options);
      
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
      
      String fieldName = cmd.getOptionValue(FIELD_NAME_PARAM);
      if (fieldName == null) {
        Usage("Specify: '" + FIELD_NAME_DESC, options);
      }
      
      String [] subDirs = subDirTypeList.split(",");
      
      int fieldId = -1;
      for (int k = 0; k < FeatureExtractor.mFieldNames.length; ++k) {
        if (FeatureExtractor.mFieldNames[k].equalsIgnoreCase(fieldName)) {
          fieldId = k;
          break;
        }
      }
      
      if (fieldId == -1) {
        System.err.println("Wrong field name: '" + fieldName + "' available fields:");
        for (int k = 0; k < FeatureExtractor.mFieldNames.length; ++k)
          System.err.print(FeatureExtractor.mFieldNames[k] + " ");
        Usage("Specify: '" + FIELD_NAME_DESC, options);
      }
        
      String fieldSOLR = FeatureExtractor.mFieldsSOLR[fieldId];
      
      System.out.println("Processing field: '" + fieldName + 
                         "' the name of the field in the SOLR index-file: '" + fieldSOLR + "'");
        
      String [] fileNames = new String[subDirs.length];
      for (int i = 0; i < fileNames.length; ++i)
        fileNames[i] = rootDir + "/" + subDirs[i] + "/" + solrFileName;
        
      InMemForwardIndex indx = new InMemForwardIndex(fieldSOLR, fileNames, maxNumRec);
        
      indx.save(InMemIndexFeatureExtractor.indexFileName(outPrefix, fieldName));
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
  }

}
