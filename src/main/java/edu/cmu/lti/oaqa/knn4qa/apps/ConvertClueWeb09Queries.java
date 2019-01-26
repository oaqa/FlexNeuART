/*
 *  Copyright 2019 Carnegie Mellon University
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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;


/**
 * A class that converts ClueWeb09 queries to our internal XML format. 
 * The query is stemmed and copied to every field.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertClueWeb09Queries {
  
  private static final XmlHelper mXmlHlp = new XmlHelper();

  public static final class Args {
    @Option(name = "-" + CommonParams.SOLR_FILE_NAME_PARAM, required = true, usage = CommonParams.SOLR_FILE_NAME_DESC)
    String mSolrFileName;
    
    @Option(name = "-in_file", required = true, usage = "Input file")
    String mInFile;

    @Option(name = "-" + ConvertClueWeb09.STOP_WORD_FILE, required = true, usage = ConvertClueWeb09.STOP_WORD_FILE_DESC)
    String mStopWordFile;
    
    @Option(name = "-" + ConvertClueWeb09.COMMON_WORD_FILE, required = true, usage = ConvertClueWeb09.COMMON_WORD_FILE_DESC)
    String mCommonWordFile;
  }
  
  public static void main(String[] argv) {
    Args args = new Args();
    CmdLineParser parser = null;
    
    try {
 
      parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(CommonParams.USAGE_WIDTH));
      parser.parseArgument(argv);
    
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.exit(1);
    }
    
    BufferedReader inQueryFile = null;
    BufferedWriter outQueryFile = null;
    
    try {
      ClueWeb09TextProc textProc = new ClueWeb09TextProc(args.mStopWordFile, 
                                                         args.mCommonWordFile, 
                                                         ConvertClueWeb09.LOWERCASE);
      
      inQueryFile = new BufferedReader(new InputStreamReader(new FileInputStream(args.mInFile)));
      outQueryFile = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args.mSolrFileName)));   
      
      String line;
      while ((line = inQueryFile.readLine()) != null) {
        if (line.isEmpty()) {
          continue;
        }
        int icol = line.indexOf(':');
        if (icol < 0) {
          System.out.println("Malformed query: " + line);
          System.exit(1);
        }
        String qid = line.substring(0, icol);
        String qtext = textProc.stemText(textProc.filterText(line.substring(icol + 1)));
        
        Map<String, String>  fieldInfo = new HashMap<String, String>();
        
        fieldInfo.put(UtilConst.TAG_DOCNO, qid);
        for (String fn : ConvertClueWeb09.getStemmedFieldNames()) {
          fieldInfo.put(fn, qtext);
        }
        
        outQueryFile.write(mXmlHlp.genXMLIndexEntry(fieldInfo));
        outQueryFile.write(UtilConst.NL);    
      }
      
    } catch (Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } finally {
      try {
        inQueryFile.close();
        outQueryFile.close();
      } catch (IOException e) {
        e.printStackTrace();
        System.exit(1);
      }
     
    }  

  }

}
