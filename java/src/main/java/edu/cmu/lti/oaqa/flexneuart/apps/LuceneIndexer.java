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

import org.apache.commons.cli.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;

import java.io.*;
import java.nio.file.Paths;

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
  final static Logger logger = LoggerFactory.getLogger(LuceneIndexer.class);
  
  public static final String EXACT_MATCH_PARAM = "exact_match";
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "LuceneIndexer", opt);      
    System.exit(1);

  }  
  
  public static final FieldType FULL_TEXT_FIELD_TYPE = new FieldType();
  
  /* A full-text search field: tokenized, indexed, but not stored. */
  static {
    FULL_TEXT_FIELD_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    FULL_TEXT_FIELD_TYPE.setTokenized(true);
    FULL_TEXT_FIELD_TYPE.setStored(false);
    FULL_TEXT_FIELD_TYPE.freeze();
  }
  
  public static void main(String [] args) {
    
    Options options = new Options();
    
    options.addOption(CommonParams.INPUT_DATA_DIR_PARAM,null, true, CommonParams.INPUT_DATA_DIR_DESC);
    options.addOption(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM,  null, true, CommonParams.INPDATA_SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,   null, true, CommonParams.MAX_NUM_REC_DESC); 
    options.addOption(CommonParams.DATA_FILE_PARAM,     null, true, CommonParams.DATA_FILE_DESC);
    options.addOption(CommonParams.OUT_INDEX_PARAM,     null, true, CommonParams.OUT_INDEX_DESC);
    options.addOption(CommonParams.INDEX_FIELD_NAME_PARAM, null, true, CommonParams.INDEX_FIELD_NAME_DESC);
    options.addOption(EXACT_MATCH_PARAM, null, false, "Create index for exact search");
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      boolean exactMatch = cmd.hasOption(EXACT_MATCH_PARAM);
      
      String indexFieldName = cmd.getOptionValue(CommonParams.INDEX_FIELD_NAME_PARAM);
      
      if (indexFieldName == null) Usage("Specify: " + CommonParams.INDEX_FIELD_NAME_DESC, options);
      
      String inputDataDir = null;
      
      inputDataDir = cmd.getOptionValue(CommonParams.INPUT_DATA_DIR_PARAM);
      
      if (null == inputDataDir) Usage("Specify: " + CommonParams.INPUT_DATA_DIR_PARAM, options);
      
      String outputDirName = cmd.getOptionValue(CommonParams.OUT_INDEX_PARAM);
      
      if (null == outputDirName) Usage("Specify: " + CommonParams.OUT_INDEX_PARAM, options);
      
      String subDirList = cmd.getOptionValue(CommonParams.INPDATA_SUB_DIR_TYPE_PARAM);
      
      if (null == subDirList ||
          subDirList.isEmpty()) Usage("Specify: " + CommonParams.INPDATA_SUB_DIR_TYPE_PARAM, options);
      
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

      createLuceneIndex(inputDataDir, subDirList, dataFileName, outputDirName, indexFieldName, exactMatch, maxNumRec);
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
    	e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
  }

  public static void createLuceneIndex(String inputDataDir, 
		  								String subDirList, String dataFileName,
		  								String outputDirName, 
		  								String indexFieldName, boolean exactMatch, 
		  								int maxNumRec) throws IOException, Exception {
	  File outputDir = new File(outputDirName);
	  if (!outputDir.exists()) {
		  if (!outputDir.mkdirs()) {
			  System.err.println("couldn't create " + outputDir.getAbsolutePath());
			  System.exit(1);
		  }
	  }
	  if (!outputDir.isDirectory()) {
		  System.err.println(outputDir.getAbsolutePath() + " is not a directory!");
		  System.exit(1);
	  }
	  if (!outputDir.canWrite()) {
		  System.err.println("Can't write to " + outputDir.getAbsolutePath());
		  System.exit(1);
	  }

	  String subDirs[] = subDirList.split(",");

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
	  indexConf.setRAMBufferSizeMB(LuceneCandidateProvider.DEFAULT_RAM_BUFFER_SIZE);
	  logger.info("Creating a new Lucene index, maximum # of docs to process: " + maxNumRec + 
			  " index field name: " + indexFieldName + 
			  " exact match? " + exactMatch);
	  indexConf.setOpenMode(OpenMode.CREATE);
	  IndexWriter indexWriter = new IndexWriter(indexDir, indexConf);      

	  for (int subDirId = 0; subDirId < subDirs.length && docNum < maxNumRec; ++subDirId) {
		  String inputFileName = inputDataDir + File.separator + subDirs[subDirId] + File.separator + dataFileName;

		  logger.info("Input file name: " + inputFileName);   
		  try (DataEntryReader inp = new DataEntryReader(inputFileName)) {
			  DataEntryFields docFields = null;

			  for (; ((docFields = inp.readNext()) != null) && docNum < maxNumRec; ) {
				  ++docNum;

				  String id = docFields.mEntryId;

				  if (id == null) {
					  logger.warn("Ignoring document #" + docNum + " b/c it has no document ID.");
					  continue;
				  }

				  String textFieldValue = docFields.getString(indexFieldName);

				  if (textFieldValue == null) {
					  logger.warn(String.format("Warning: No field '%s', offending DOC #%d", 
							  indexFieldName, docNum));
					  continue;
				  }

				  Document luceneDoc = new Document();
				  luceneDoc.add(new StringField(Const.DOC_ID_FIELD_NAME, id, Field.Store.YES));

				  if (exactMatch) {
					  luceneDoc.add(new StringField(indexFieldName, textFieldValue, Field.Store.NO));
				  } else { 
					  luceneDoc.add(new Field(indexFieldName, textFieldValue, FULL_TEXT_FIELD_TYPE));
				  }

				  indexWriter.addDocument(luceneDoc);
				  if (docNum % Const.PROGRESS_REPORT_QTY == 0) {
					  logger.info("Indexed " + docNum + " docs");
				  }

			  }
			  logger.info("Indexed " + docNum + " docs");
		  }
	  }

	  indexWriter.commit();
	  indexWriter.close();
	}
}
