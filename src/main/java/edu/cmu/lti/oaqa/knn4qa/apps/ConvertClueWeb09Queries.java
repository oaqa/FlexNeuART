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

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

/**
 * A class that converts ClueWeb09 queries to our internal XML format. 
 * The query is stemmed and copied to every field.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertClueWeb09Queries {

  public static final class Args {
    @Option(name = "-" + CommonParams.SOLR_FILE_NAME_PARAM, required = true, usage = CommonParams.SOLR_FILE_NAME_DESC)
    String mSolrFileName;
    
    @Option(name = "-in_file", required = true, usage = "Input file")
    String mInFile;
    
    @Option(name = "-out_file", required = true, usage = "Input file")
    String mOutFile;
    
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
    
    

  }

}
