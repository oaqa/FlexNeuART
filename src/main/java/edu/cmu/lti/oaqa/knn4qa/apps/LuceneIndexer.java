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

import org.apache.commons.cli.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.*;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.LuceneCandidateProvider;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

/**
 * An simple application that reads text files produced by an annotation
 * pipeline and indexes their content using SOLR.
 * <p>
 * <b>A limitation: can index only text fields, the ID field name is hardcoded.</b>
 * <p>
 * The input file contains documents. The document
 * text is enclosed between tags &lt;DOC&gt; and &lt;DOC&gt; and occupies
 * exactly one line.
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
  
  public static void main(String [] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.ROOT_DIR_PARAM,      null, true, CommonParams.ROOT_DIR_DESC);
    options.addOption(CommonParams.SUB_DIR_TYPE_PARAM,  null, true, CommonParams.SUB_DIR_TYPE_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM,   null, true, CommonParams.MAX_NUM_REC_DESC);
    options.addOption(CommonParams.SOLR_FILE_NAME_PARAM,null, true, CommonParams.SOLR_FILE_NAME_DESC);    
    options.addOption(CommonParams.OUT_INDEX_PARAM,     null, true, CommonParams.OUT_MINDEX_DESC);    

    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String rootDir = null;
      
      rootDir = cmd.getOptionValue(CommonParams.ROOT_DIR_PARAM);
      
      if (null == rootDir) Usage("Specify: " + CommonParams.ROOT_DIR_DESC, options);
      
      String outputDirName = cmd.getOptionValue(CommonParams.OUT_INDEX_PARAM);
      
      if (null == outputDirName) Usage("Specify: " + CommonParams.OUT_MINDEX_DESC, options);
      
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
        String inputFileName = rootDir + "/" + subDirs[subDirId] + "/" + solrFileName;
        
        System.out.println("Input file name: " + inputFileName);        

        BufferedReader inpText = new BufferedReader(new InputStreamReader(
            CompressUtils.createInputStream(inputFileName)));
        String docText = XmlHelper.readNextXMLIndexEntry(inpText);

        for (; docText != null && docNum < maxNumRec; docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
          ++docNum;
          Map<String, String> docFields = null;

          Document luceneDoc = new Document();

          try {
            docFields = XmlHelper.parseXMLIndexEntry(docText);
          } catch (Exception e) {
            System.err.println(String.format("Parsing error, offending DOC #%d:\n%s", docNum, docText));
            System.exit(1);
          }

          String id = docFields.get(UtilConst.TAG_DOCNO);

          if (id == null) {
            System.err.println(String.format("No ID tag '%s', offending DOC #%d:\n%s", UtilConst.TAG_DOCNO, docNum,
                docText));
          }

          luceneDoc.add(new StringField(UtilConst.TAG_DOCNO, id, Field.Store.YES));

          for (Map.Entry<String, String> e : docFields.entrySet())
            if (!e.getKey().equals(UtilConst.TAG_DOCNO)) {
              luceneDoc.add(new TextField(e.getKey(), e.getValue(), Field.Store.YES));
            }
          indexWriter.addDocument(luceneDoc);
          if (docNum % 1000 == 0) {
            System.out.println("Indexed " + docNum + " docs");
          }
          if (docNum % COMMIT_INTERV == 0) {
            System.out.println("Committing");
            indexWriter.commit();
          }
        }
        System.out.println("Indexed " + docNum + " docs");
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
