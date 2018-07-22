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

import java.io.*;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;

/**
 *  An application to convert output of our pipeline to TREC text format (only for the field text)
 *  
 * @author Leoind Boytsov
 *
 */
public class Convert2TrecText {
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(Convert2TrecText.class.getName(), opt);      
    System.exit(1);
  }  
  
  public static void main(String[] args) {
    Options options = new Options();

    options.addOption("input_file",      null, true, 
                      "Input file names (can be several options) each of which is an XML produced by our pipeline");
    options.addOption("output_file",     null, true,
                      "Output file name (in the TREC text format)");
    options.addOption("batch_qty",       null, true,
                      "Number of documents in a batch");
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    BufferedWriter    outFile = null;
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String [] inpFileNames = cmd.getOptionValues("input_file");
      if (inpFileNames == null || inpFileNames.length == 0) {
        Usage("Specify one or more input file!", options);
      }
      String outFileName = cmd.getOptionValue("output_file");
      if (null == outFileName) {
        Usage("Specify the name of an output file!", options);
      }
      String tmpi = cmd.getOptionValue("batch_qty");
      int    batchQty = -1;
      
      if (null != tmpi) {
        batchQty = Integer.parseInt(tmpi);
      }
      if (batchQty <= 0) {
        Usage("Specify a positive # of documents in a batch", options);
      }
      
      int batchNum = 1;
      int qty = 0;
      
      outFile = initOutFile(outFile, outFileName, batchNum);
      
      String textFieldName = FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_FIELD_ID];
      
      for (String inpFileName : inpFileNames) {
        System.out.println("Started to process file: " + inpFileName);
        BufferedReader inpText = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inpFileName)));
        
        String docText = XmlHelper.readNextXMLIndexEntry(inpText);
        int docNum = 0;
        for (; docText != null; docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
          ++docNum; ++qty;
          if (qty >= batchQty) {
            ++batchNum;
            outFile = initOutFile(outFile, outFileName, batchNum);
            qty = 0;
          }
          Map<String, String> docFields = null;
          
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
      
          String text = docFields.get(textFieldName);
          if (text != null && !text.isEmpty()) {
            outFile.write(String.format("<DOC>\n<DOCNO>%s</DOCNO>\n<TEXT>\n%s\n</TEXT>\n</DOC>\n", id, text));
          } else {
            System.err.println(String.format("Warning: empty text field for id=%s", id));
          }
        }
        System.out.println("Processed " + docNum + " records from file: " + inpFileName);
      }
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }
    
    if (outFile != null) {
      try {
        outFile.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    
  }

  private static BufferedWriter initOutFile(BufferedWriter outFile, String outFileName, int batchNum) throws IOException {
    if (outFile!=null) outFile.close();
    String fn = outFileName + "." + batchNum + ".trectext";
    System.out.println("Creating a file: " + fn);
    return new BufferedWriter(new FileWriter(new File(fn)));
  }

}
