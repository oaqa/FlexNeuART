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
import java.util.ArrayList;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.squad.*;

public class WikiTextConverter {
  private static final String OUTPUT_FILE = "output_file";
  private static final String INPUT_SQUAD_DIR = "input_squad_dir";
  private static final String INPUT_WIKI_DIR = "input_wiki_dir";

  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(WikiTextConverter.class.getCanonicalName(), mOptions);      
    System.exit(1);
  }
  
  public static Options mOptions = new Options();;
  
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
      
      SQuADWikiTitlesReader wikiTitles = new SQuADWikiTitlesReader(inputSQuADDir);
      
      ArrayList<Path> wikiFiles = new ArrayList<Path>();
      
      collectFiles(new File(inputWikiDir).toPath(), wikiFiles);
      
      System.out.println("Input Wiki files:");
      
      for (Path p : wikiFiles)
        System.out.println(p);
         
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
    
}
