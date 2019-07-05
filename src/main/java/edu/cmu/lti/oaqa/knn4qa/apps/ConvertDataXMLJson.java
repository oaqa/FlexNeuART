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


import java.util.Map;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryWriter;

/**
 * A helper class to convert from XML-per-entry to JSONL format and vice versa.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertDataXMLJson {
  public static final class Args {  
    @Option(name = "-input", required = true, usage = "input file")
    String mInputFileName;
    @Option(name = "-output", required = true, usage = "output file")
    String mOutputFileName;
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
    
    try {
      
      try (DataEntryWriter out = new DataEntryWriter(args.mOutputFileName)) {
        try (DataEntryReader inp = new DataEntryReader(args.mInputFileName)) {
          Map<String, String> docFields = null;   
          
          while ((docFields = inp.readNext()) != null) {
            out.writeEntry(docFields);
          }
        }
        
      }
      
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Exception while processing: " + e);
      System.exit(1);
    }

  }

}
