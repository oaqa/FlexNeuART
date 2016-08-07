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

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.ParsedQuestion;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersParser;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersStreamParser;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;


/**
 *  Just a simple class to test if the Yahoo parsing function that works on
 *  top of an XML iterator is fine.
 */
public class TestXmlIterYahooParserApp {
  
  private static final String NL = System.getProperty("line.separator");

  public static void usage(String err) {
    System.err.println("Error: " + err);
    System.err.println("Usage: <input file> <do clean up?>");
    System.exit(1);
  }

  public static void main(String[] args) {
    if (args.length != 2) usage("Wrong number of arguments");
    String inputFileName = args[0];
    boolean doCleanUp = Boolean.parseBoolean(args[1]);
    
    System.out.println("Do clean up: " + doCleanUp);

    try {
      XmlIterator               xmlIter = new XmlIterator(CompressUtils.createInputStream(inputFileName), "document");
      YahooAnswersStreamParser  xmlStream = new YahooAnswersStreamParser(inputFileName, doCleanUp);
      
      String xmlIterStr; 
      int recNum = 0;
      int diffQty = 0;
      
      while (true) {
        xmlIterStr = xmlIter.readNext();
        
        ParsedQuestion q1 = null, q2 = null; 
        ++recNum;

        if (!xmlIterStr.isEmpty()) q1 = YahooAnswersParser.parse(xmlIterStr, doCleanUp);
        if (xmlStream.hasNext())   q2 = xmlStream.next();

        
        if (q1 == null && q2 == null) break;
        if (q1 == null && q2 != null) {
          System.err.println(String.format(
                          "Mismatch in record %d, q1 == null, q2 != null", 
                          recNum));
          System.exit(1);
        }
        if (q1 != null && q2 == null) {
          System.err.println(String.format(
                          "Mismatch in record %d, q1 != null, q2 == null", 
                          recNum));
          System.exit(1);
        }
        
        boolean b = q1.compare(q2, true);
        
        //System.out.println(q1);
        
        if (!b) {
          System.err.println(String.format("Mismatch in record %d",  recNum));
          System.err.println(String.format("q1:\n%s", q1));
          System.err.println("==================================");
          System.err.println(String.format("q2:\n%s", q2));
          diffQty++;
          //System.exit(1);
        }
        if (recNum % 10000 == 0) System.out.println(String.format("# of rec processed: %d, # of mismatches %d", 
                                                                  recNum, diffQty));
      }
      System.out.println(String.format("# of rec processed: %d, # of mismatches %d", 
                                       recNum, diffQty));
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      System.exit(1);
    }
  }
}
