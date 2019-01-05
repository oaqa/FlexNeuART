package edu.cmu.lti.oaqa.knn4qa.apps;
/*
 *  Copyright 2018 Carnegie Mellon University
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


import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import java.net.URI;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;
import org.lemurproject.kstem.KrovetzStemmer;

import edu.cmu.lemurproject.WarcRecord;
import edu.cmu.lemurproject.WarcHTMLResponseRecord;
import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.knn4qa.utils.DictNoComments;
import edu.cmu.lti.oaqa.knn4qa.utils.HtmlDocData;
import edu.cmu.lti.oaqa.knn4qa.utils.LeoHTMLParser;

class WarcProcessor implements Runnable  {
  final String warcRecFileName;
  final int expQty;
  
  WarcProcessor(String fileName, int expQty) {
    warcRecFileName = fileName;
    this.expQty = expQty;
  }
  
  @Override
  public void run() {
    
    System.out.println("Started processing file: " + warcRecFileName + " expecting: " + expQty + " records");
    
    int procQty = 0;
    try {
      DataInputStream inpWarc = new DataInputStream(CompressUtils.createInputStream(warcRecFileName));
      WarcRecord currRec = null;
      
      XmlHelper xmlHlp = new XmlHelper();
      
      while ((currRec = WarcRecord.readNextWarcRecord(inpWarc)) != null) {
        
        if (currRec.getHeaderRecordType().equals("response")) {
          WarcHTMLResponseRecord wResp = new WarcHTMLResponseRecord(currRec);
          
          String id = wResp.getTargetTrecID();
          String baseHref = ConvertClueWeb09.normalizeURL(wResp.getTargetURI());
          String response = wResp.getRawRecord().getContentUTF8();
          int endOfHead = response.indexOf("\n\n");
          if (endOfHead >= 0) {
            String html = response.substring(endOfHead + 2);
         
            HtmlDocData htmlData = LeoHTMLParser.parse(ConvertClueWeb09.ENCODING, baseHref, html);

            Map<String, String>  fieldInfo = new HashMap<String, String>();

            
            String titleUnlemm = ConvertClueWeb09.filterText(htmlData.mTitle);
            String title = ConvertClueWeb09.stemText(titleUnlemm);
            
            String bodyTextUnlemm = ConvertClueWeb09.filterText(htmlData.mBodyText);
            String bodyText = ConvertClueWeb09.stemText(bodyTextUnlemm);
            
            String linkTextUnlemm = ConvertClueWeb09.filterText(htmlData.mLinkText);
            String linkText = ConvertClueWeb09.stemText(linkTextUnlemm);


            fieldInfo.put(UtilConst.TAG_DOCNO, id);
            
            fieldInfo.put(ConvertClueWeb09.TITLE_FIELD_NAME, title);
            //fieldInfo.put(TITLE_UNLEMM_FIELD_NAME, titleUnlemm);
            
            fieldInfo.put(CandidateProvider.TEXT_FIELD_NAME, bodyText);
            //fieldInfo.put(CandidateProvider.TEXT_UNLEMM_FIELD_NAME, allTextUnlemm);
            
            fieldInfo.put(ConvertClueWeb09.LINK_TEXT_FIELD_NAME, linkText);
            
            ConvertClueWeb09.writeEntry(xmlHlp.genXMLIndexEntry(fieldInfo));
            procQty++;
          }
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    System.out.println("Finished processing file: " + warcRecFileName + 
                        " expected: " + expQty + " recs" + 
                        " processed: " + procQty + " recs");
    if (procQty != expQty) {
      System.out.println("Record # mismatch: the number of processed records != the number of declared records!");
    }

  }
}

/**
 * An app class to convert ClueWeb to knn4qa multi-field XML files.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertClueWeb09 {
  private static final String SPACE_REGEXP_SEP = "\\s+";
  private static final String COMMON_WORD_FILE = "common_word_file";
  private static final String STOP_WORD_FILE = "stop_word_file";
  private static final String CLUEWEB09_DIR = "clueweb09_dir";
  private static final String REC_START = "*./";
  
  protected static final String ENCODING = "UTF8";
  protected static final Boolean LOWERCASE = true;
  
  static protected String TITLE_FIELD_NAME = "title";
  static protected final String TITLE_UNLEMM_FIELD_NAME = "title_unlemm";
  static protected final String LINK_TEXT_FIELD_NAME = "linkText";
  
  protected static final String NL = System.getProperty("line.separator");
  
  private static KrovetzStemmer mStemmer = new KrovetzStemmer();
  private static DictNoComments mStopWords = null;
  private static DictNoComments mCommonWords = null;

  protected static BufferedWriter mOutFile;
  
  /**
   * Remove stop words and keep only words from a list.
   * If the LOWERCASE is true, then all words are lower-cased.
   * 
   * @param text
   * @return
   */
  static String filterText(String text) {
    ArrayList<String> res = new ArrayList<String>();
    
    for (String tok : text.split(SPACE_REGEXP_SEP)) {
      if (LOWERCASE) tok = tok.toLowerCase();
      
      if (!mStopWords.contains(tok) && mCommonWords.contains(tok)) {
        res.add(tok);
      }
    }
    return String.join(" ", res);
  }
  
  protected static synchronized void writeEntry(String e) throws IOException {
    mOutFile.write(e);
    mOutFile.write(NL);
  }

  static String stemText(String text) {
    String toks[] = text.split(SPACE_REGEXP_SEP);
 
    for (int i = 0; i < toks.length; ++i) {
      String s = mStemmer.stem(toks[i]);
      toks[i] = s;
    }
    return String.join(" ", toks);
  }

  public static String normalizeURL(String URL) {
    URI   uri;
    try {
      uri = new URI(URL);
    } catch (Exception e) {
      return URL.trim();
    }
    String host   = uri.getHost();
    String scheme = uri.getScheme();

    if (host == null || scheme == null ||
       (!scheme.equals("http") && !scheme.equals("https") && !scheme.equals("ftp"))) {
      return URL.trim();
    }

    String Path = uri.getPath();

    if (Path == null || Path.isEmpty()) {
      Path = "/";
    }

    try {
      uri = new URI(scheme, null /* user info */, host, uri.getPort(), Path, null /* query */, null /* fragment */);
    } catch (Exception e) {
      return URL.trim();
    };

    return uri.toString().trim();
  }
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ConvertClueWeb09", opt);      
    System.exit(1);

  } 
  
  //CandidateProvider.TEXT_FIELD_NAME
  //CandidateProvider.TEXT_UNLEMM_FIELD_NAME
  public static void main(String args[]) {
    Options options = new Options();
    
    options.addOption(CLUEWEB09_DIR,  null, true, "A root ClueWeb09 directory");
    options.addOption(CommonParams.SOLR_FILE_NAME_PARAM,null, true, CommonParams.SOLR_FILE_NAME_DESC); 
    options.addOption(STOP_WORD_FILE, null, true, "A list of stop words");
    options.addOption(COMMON_WORD_FILE, null, true, "A list of common words to restrict a set of words added to the index");
    options.addOption(CommonParams.THREAD_QTY_PARAM, null, true, CommonParams.THREAD_QTY_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    mOutFile = null;
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String solrFileName = cmd.getOptionValue(CommonParams.SOLR_FILE_NAME_PARAM);
      if (null == solrFileName) Usage("Specify: " + CommonParams.SOLR_FILE_NAME_DESC, options);
      
      String stopWordFile = cmd.getOptionValue(STOP_WORD_FILE);
      if (null == stopWordFile)  Usage("Specify: " + STOP_WORD_FILE, options);
      
      mStopWords = new DictNoComments(new File(stopWordFile), LOWERCASE);
      
      String commonWordFile = cmd.getOptionValue(COMMON_WORD_FILE);
      if (null == commonWordFile) Usage("Specify: " + COMMON_WORD_FILE, options);
      
      mCommonWords = new DictNoComments(new File(commonWordFile), LOWERCASE);
      
      System.out.println("# of common words to use:" + mCommonWords.getQty());
      System.out.println("# of stop words to use:" + mStopWords.getQty());
      
      String tmpn = cmd.getOptionValue(CommonParams.THREAD_QTY_PARAM);
      int threadQty = Runtime.getRuntime().availableProcessors();
      if (null != tmpn)
        threadQty = Integer.parseInt(tmpn);
      
      System.out.println("Using # of threads: " + threadQty);
      
      mOutFile = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(solrFileName)));
      
      String clueWebDir = cmd.getOptionValue(CLUEWEB09_DIR);
      if (null == clueWebDir) Usage("Specify: " + CLUEWEB09_DIR, options);
      
      String recFileName = clueWebDir + "/record_counts/ClueWeb09_English_1_counts.txt";
      
      ExecutorService executor = Executors.newFixedThreadPool(threadQty);
        
      for (String line: FileUtils.readLines(new File(recFileName))) {
        String tmp[] = line.split(SPACE_REGEXP_SEP);
        String firstPart = tmp[0];
        int expQty = Integer.parseInt(tmp[1]);

        if (firstPart.trim().isEmpty()) {
          continue;
        }
        if (firstPart.startsWith(REC_START)) {
          String warcRecFileName = clueWebDir + "/ClueWeb09_English_1/" + firstPart.substring(REC_START.length());
          
          executor.execute(new WarcProcessor(warcRecFileName, expQty));
          
        } else {
          System.err.println(String.format("Invalid line %s in file %s", line, recFileName));
          System.exit(1);
        }
      }
                
      executor.shutdown();
      executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

    
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } finally {
      if (mOutFile != null) {
        try {
          mOutFile.close();
        } catch (IOException e) {
          e.printStackTrace();
          System.err.println("Error closing output stream!");
          System.exit(1);
        }
      }
    }
    
    
  }
}
