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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.*;

import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

/**
 * An simple application that reads text files produced by an annotation
 * pipeline and indexes their content using Lucene. 
 * 
 * Limitations: the name of ID-field and indexable text fields are hard-coded.
 * 
 * @author Leonid Boytsov
 *
 */

public class LuceneIndexer {
  public static final int COMMIT_INTERV = 50000;
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "LuceneIndexer", opt);      
    System.exit(1);

  }  
  
  
  public static final FieldType FIELD_TYPE = new FieldType();
  
  /* Indexed, tokenized, not stored. */
  static {
    FIELD_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    FIELD_TYPE.setTokenized(true);
    FIELD_TYPE.setStored(false);
    FIELD_TYPE.freeze();
  }
  
  public static void main(String [] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.INPUT_DATA_DIR_PARAM,null, true, CommonParams.INPUT_DATA_DIR_DESC);
    options.addOption(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM,  null, true, CommonParams.INPDATA_SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,   null, true, CommonParams.MAX_NUM_REC_DESC); 
    options.addOption(CommonParams.DATA_FILE_PARAM,     null, true, CommonParams.DATA_FILE_DESC);
    options.addOption(CommonParams.OUT_INDEX_PARAM,     null, true, CommonParams.OUT_INDEX_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String inputDataDir = null;
      
      inputDataDir = cmd.getOptionValue(CommonParams.INPUT_DATA_DIR_PARAM);
      
      if (null == inputDataDir) Usage("Specify: " + CommonParams.INPUT_DATA_DIR_PARAM, options);
      
      String outputDirName = cmd.getOptionValue(CommonParams.OUT_INDEX_PARAM);
      
      if (null == outputDirName) Usage("Specify: " + CommonParams.OUT_INDEX_PARAM, options);
      
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

      File outputDir = new File(outputDirName);
      if (!outputDir.exists()) {
        if (!outputDir.mkdirs()) {
          System.out.println("couldn't create " + outputDir.getAbsolutePath());
          System.exit(1);
        }
      }
      if (!outputDir.isDirectory()) {
        System.out.println(outputDir.getAbsolutePath() + " is not a directory!");
        System.exit(1);
      }
      if (!outputDir.canWrite()) {
        System.out.println("Can't write to " + outputDir.getAbsolutePath());
        System.exit(1);
      }
      
      String subDirs[] = subDirTypeList.split(",");

      int docNum = 0;

      // No English analyzer here, all language-related processing is done already,
      // here we simply white-space tokenize and index tokens verbatim.
      Analyzer analyzer = new WhitespaceAnalyzer();
      FSDirectory       indexDir    = FSDirectory.open(Paths.get(outputDirName));
      IndexWriterConfig indexConf   = new IndexWriterConfig(analyzer);
      
      /*
          OpenMode.CREATE creates a new index or overwrites an existing one.
          https://lucene.apache.org/core/6_0_0/core/org/apache/lucene/index/IndexWriterConfig.OpenMode.html#CREATE
      */
      indexConf.setOpenMode(OpenMode.CREATE); 
      indexConf.setRAMBufferSizeMB(LuceneCandidateProvider.RAM_BUFFER_SIZE);
      System.out.println("Creating a new Lucene index, maximum # of docs to process: " + maxNumRec);
      indexConf.setOpenMode(OpenMode.CREATE);
      IndexWriter indexWriter = new IndexWriter(indexDir, indexConf);      
      
      for (int subDirId = 0; subDirId < subDirs.length && docNum < maxNumRec; ++subDirId) {
        String inputFileName = inputDataDir + File.separator + subDirs[subDirId] + File.separator + dataFileName;
        
        System.out.println("Input file name: " + inputFileName);   
        try (DataEntryReader inp = new DataEntryReader(inputFileName)) {
          Map<String, String> docFields = null;
          
          for (; ((docFields = inp.readNext()) != null) && docNum < maxNumRec; ) {
            ++docNum;
            
            String id = docFields.get(Const.TAG_DOCNO);

            if (id == null) {
              System.out.println(String.format("Warning: No ID tag '%s', offending DOC #%d", Const.TAG_DOCNO, docNum));
              continue;
            }
            
            String textFieldName = Const.TEXT_FIELD_NAME; 
            String textFieldValue = docFields.get(textFieldName);
            
            if (textFieldValue == null) {
              System.out.println(String.format("Warning: No field '%s', offending DOC #%d", 
                                               textFieldName, docNum));
              continue;
            }

            Document luceneDoc = new Document();
            luceneDoc.add(new StringField(Const.TAG_DOCNO, id, Field.Store.YES));
         
            luceneDoc.add(new Field(textFieldName, textFieldValue, FIELD_TYPE));
            
            indexWriter.addDocument(luceneDoc);
            if (docNum % Const.PROGRESS_REPORT_QTY == 0) {
              System.out.println("Indexed " + docNum + " docs");
            }
            if (docNum % COMMIT_INTERV == 0) {
              System.out.println("Committing");
              indexWriter.commit();
            }
          }
          System.out.println("Indexed " + docNum + " docs");
        }
      }
      
      indexWriter.commit();
      indexWriter.close();
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
  }
}
