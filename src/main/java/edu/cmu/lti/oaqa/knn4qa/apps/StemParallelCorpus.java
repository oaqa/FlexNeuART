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
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.solr.UtilConst;

/**
 * 
 * A class producing a stemmed version of the unlemmatized parallel corpus.
 * 
 * @author Leonid Boytsov
 *
 */
public class StemParallelCorpus {
  public static final class Args {
    @Option(name = "-srcDir", required = true, usage = "A source root directory")
    String mSrcDir;
    
    @Option(name = "-dstDir", required = true, usage = "A target root directory")
    String mDstDir;    
    
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
    
    BufferedReader questionReader = null;
    BufferedReader answerReader = null;
    
    BufferedWriter questionWriter = null;
    BufferedWriter answerWriter = null;
    
    try {
      
      ClueWeb09TextProc textProc = new ClueWeb09TextProc(args.mStopWordFile, 
                                                         args.mCommonWordFile, 
                                                         ConvertClueWeb09.LOWERCASE);
      
      questionReader = new BufferedReader(new FileReader(new File(args.mSrcDir + "/question_text_unlemm")));
      answerReader = new BufferedReader(new FileReader(new File(args.mSrcDir + "/answer_text_unlemm")));
      
      answerWriter = new BufferedWriter(new FileWriter(new File(args.mDstDir + "/answer_text")));
      questionWriter = new BufferedWriter(new FileWriter(new File(args.mDstDir + "/question_text")));
      
      String lineQuestion = null, lineAnswer = null;

      while (true) {
        lineQuestion = questionReader.readLine();
        lineAnswer = answerReader.readLine();
        if (lineQuestion == null || lineAnswer == null)
          break;
        questionWriter.write(textProc.stemText(textProc.filterText(lineQuestion)) + UtilConst.NL);
        answerWriter.write(textProc.stemText(textProc.filterText(lineAnswer)) + UtilConst.NL);
      }
      if (lineAnswer != null) {
        throw new Exception("Error: Answer file has more entries!");
      }
      if (lineQuestion != null) {
        throw new Exception("Error: Question file has more entries!");
      }
      
    } catch (Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } finally {
      try {
        if (questionWriter != null) 
          questionWriter.close();
        if (answerWriter != null) 
          answerWriter.close();
      } catch (IOException e) {
        e.printStackTrace();
        System.exit(1);
      }
    }
  }
}
