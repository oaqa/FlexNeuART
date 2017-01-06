/*
 *  Copyright 2017 Carnegie Mellon University
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

/**
 * This class converts a textual output of Wikipedia to an intermediate JSON format.
 * <p>We assume the following:</p>
 * <ul>
 * <li>The output is produced by <a href="https://github.com/searchivarius/wikiextractor">Leo's modified version of wikiextrator</a>.
 * Leo's version retains paragraph separating empty lines.
 * <li>The SQuAD collection is split into several parts with hard-coded names, from which we can read the list of titles.
 * <li>We will ignore any wikipedia titles that are in the SQuAD collection.  
 * </ul>
 * 
 */
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.*;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.qaintermform.*;
import edu.cmu.lti.oaqa.knn4qa.squad.*;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import com.google.gson.Gson;

public class WikiTextConverter {
  private static final String DOC_END = "</doc>";
  private static final String OUTPUT_FILE = "output_file";
  private static final String INPUT_SQUAD_DIR = "input_squad_dir";
  private static final String INPUT_WIKI_DIR = "input_wiki_dir";

  
  private static Options    mOptions = new Options();
  
  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(WikiTextConverter.class.getCanonicalName(), mOptions);      
    System.exit(1);
  }
  
  static void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }  
  
  public final static void collectFiles(Path dir, final ArrayList<Path> files) throws IOException {
    Files.walkFileTree(dir, new SimpleFileVisitor<Path>() {
      @Override
      public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {

        if (Files.isReadable(file) && file.getFileName().toString().startsWith("wiki_")) {
          files.add(file.toRealPath());
        }
        return FileVisitResult.CONTINUE;
      }
    });
  }

  public static void main(String[] args) {
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    BufferedWriter    outFile = null;
    
    mOptions.addOption(INPUT_WIKI_DIR,      null, true, 
        "Directory contatining preprocessed Wikipedia dump (preprocessed by Leo's modification of wikiextractor (https://github.com/searchivarius/wikiextractor))");

    mOptions.addOption(INPUT_SQUAD_DIR,     null, true,
        "Directory containing split SQuAD files as well as the dev1 (we expect certain hardcoded names)");
    
    mOptions.addOption(OUTPUT_FILE,     null, true,
        "Output file name in our intermediate format");
    
    try {
      CommandLine cmd = parser.parse(mOptions, args);
   
      String inputWikiDir = cmd.getOptionValue(INPUT_WIKI_DIR);
      if (null == inputWikiDir) showUsageSpecify(INPUT_WIKI_DIR);
      String inputSQuADDir = cmd.getOptionValue(INPUT_SQUAD_DIR);
      if (inputSQuADDir == null) showUsageSpecify(INPUT_SQUAD_DIR);
      String outFileName = cmd.getOptionValue(OUTPUT_FILE);
      if (null == outFileName) showUsageSpecify(OUTPUT_FILE);
      
      outFile = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outFileName)));
      
      SQuADWikiTitlesReader wikiTitles = new SQuADWikiTitlesReader(inputSQuADDir);
      
      ArrayList<Path> wikiFiles = new ArrayList<Path>();
      
      collectFiles(new File(inputWikiDir).toPath(), wikiFiles);
      
      wikiFiles.sort(new Comparator<Path>() {
        @Override
        public int compare(Path p1, Path p2) {
          return p1.compareTo(p2);
        }
      });
      
      System.out.println("The number of input wiki files: " + wikiFiles.size());
      
      for (Path p : wikiFiles) {
        processOneFile(p.toFile(), wikiTitles, outFile);
      }
/*      
      System.out.println("Input Wiki files:");      
      for (Path p : wikiFiles)
        System.out.println(p);
*/         
    } catch (ParseException e) {
      showUsage("Cannot parse arguments");
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
  
  private static Gson mGSON = new Gson();  

  private static void processOneFile(File inpFile, SQuADWikiTitlesReader wikiTitles, BufferedWriter outFile) 
      throws Exception {
    System.out.println("Processing file: " + inpFile.getAbsolutePath());
    BufferedReader input 
    = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inpFile.getAbsolutePath())));
    
    int lineNum = 0;            
    
    while (true) {
      String s;
      
      s = input.readLine(); ++lineNum;
      if (s == null) break;
      s = s.trim();
      if (s.isEmpty()) continue;
      if (!s.startsWith("<doc"))
        throw new Exception("Expecting a start of an entry, line " + lineNum + " file: " + inpFile);
      //System.out.println("Start: " + lineNum);
      String head = s;
      
      ArrayList<String> docLines = new ArrayList<String>();
      
      while (true) {
        s = input.readLine(); ++lineNum;
        if (s == null)
          throw new Exception("Expecting an end of an entry, line " + lineNum + " file: " + inpFile);
        if (s.equals(DOC_END)) {
          //System.out.println("End: " + lineNum);
          break;
        } else {
          docLines.add(s + " "); // the space will play a role of separator\
        }
      }
      
      // At this point we should have read one complete parsed entry
      
      Document        doc  = XmlHelper.parseDocWithoutXMLDecl(head + DOC_END);
      Node            root = doc.getFirstChild();
      NamedNodeMap    attrs = root.getAttributes();
      String pageId    = attrs.getNamedItem("id").getNodeValue();
      String title = attrs.getNamedItem("title").getNodeValue();
      
      if (wikiTitles.mhTitles.contains(title)) {
        System.out.println("Ignoring a page with a title from the SQuAD collection: '" + title + "'"); 
      } else {
        ArrayList<String> docParas = new ArrayList<String>();
        String para = "";
        for (int currLine = 0; currLine < docLines.size(); ++currLine) {
          String cln = docLines.get(currLine);
          if (!cln.isEmpty()) para = para + cln;
          if (cln.isEmpty() || currLine + 1 == docLines.size()) {
            if (!para.isEmpty()) {
              docParas.add(para);
              para = "";
            }
          }
        }
        
        if (docParas.size() > 1) {     
          int       qty = docParas.size() - 1;
          QAData    data = new QAData(qty);
                             
          // We skip the first "paragraph", because it is merely a title!
          for (int inPageId = 1; inPageId < docParas.size(); ++inPageId) {
            String id = pageId + "_" + inPageId;
            data.passages[inPageId - 1] = new QAPassage(id, docParas.get(inPageId), 0 /* no questions here */);              
          }
          
          String pageText = mGSON.toJson(data, QAData.class);
       /*
        * Replace all new line characters here, so that a JSON entry is a single line
        * This is, perhaps, a bit paranoid, b/c readLine seem to remove line-ending
        * chars.
        */
          pageText = pageText.replaceAll("[\n\r]", " "); 
          outFile.write(pageText);
          outFile.write(NL);
        }
      }
      
      //System.out.println(id + " -> " + title);
      
    }
    
    input.close();    
  }  

  protected static final String NL = System.getProperty("line.separator");
}
