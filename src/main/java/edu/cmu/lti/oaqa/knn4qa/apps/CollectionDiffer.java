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

import java.io.*;
import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.collection_reader.ParsedQuestion;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersParser;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersReader;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;

/**
 * An application that "diffs" two collection:
 * it outputs all the documents whose subject doesn't coincide
 * with the subject from a document in the second collection.
 * We check only subjects, details are ignored. Thus, we might
 * delete a few extra entries.
 * 
 * @author Leonid Boytsov
 *
 */
public class CollectionDiffer {
  static void Usage(String err) {
    System.err.println("Error: " + err);
    System.err.println("Usage, to produce File1 \\ File, specify: " 
                       + "-i1 <Input file 1> "
                       + "-i2 <Input file 1> "
                       + "-o <Output file (i1 \\ i2)> "
                       );
    System.exit(1);
  }

  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption("i1",null, true, "Input file 1");
    options.addOption("i2",null, true, "Input file 2");
    options.addOption("o", null, true, "Output file");

    CommandLineParser parser = new org.apache.commons.cli.GnuParser(); 
    
   
    try {
      CommandLine cmd = parser.parse(options, args);
      
      InputStream  input1 = null, input2 = null;
      
      if (cmd.hasOption("i1")) {
        input1 = CompressUtils.createInputStream(cmd.getOptionValue("i1"));
      } else {
        Usage("Specify 'Input file 1'"); 
      }
      if (cmd.hasOption("i2")) {
        input2 = CompressUtils.createInputStream(cmd.getOptionValue("i2"));
      } else {
        Usage("Specify 'Input file 2'"); 
      }      
      
      HashSet<String>   hSubj = new HashSet<String>();
      
      BufferedWriter out = null;
      
      if (cmd.hasOption("o")) {
        String outFile = cmd.getOptionValue("o");
        
        out = new BufferedWriter(new OutputStreamWriter(
            CompressUtils.createOutputStream(outFile)));
      } else {
        Usage("Specify 'Output file'");      
      }
      
      XmlIterator   inpIter2 = new XmlIterator(input2, YahooAnswersReader.DOCUMENT_TAG);
      
      
      int docNum=1;
      for (String oneRec = inpIter2.readNext();!oneRec.isEmpty();oneRec=inpIter2.readNext(),++docNum) {
        if (docNum % 10000 == 0) {
          System.out.println(String.format("Loaded and memorized questions for %d documents from the second input file", docNum));
        }
        ParsedQuestion q = YahooAnswersParser.parse(oneRec, false);
        hSubj.add(q.mQuestion);
      }
      
      XmlIterator   inpIter1 = new XmlIterator(input1, YahooAnswersReader.DOCUMENT_TAG);
      
      System.out.println("=============================================");
      System.out.println("Memoization is done... now let's diff!!!");
      System.out.println("=============================================");
      
      docNum = 1;
      int skipOverlapQty = 0, skipErrorQty = 0;
      for (String oneRec = inpIter1.readNext(); !oneRec.isEmpty() ; ++docNum, oneRec = inpIter1.readNext()) {
        if (docNum % 10000 == 0) {
          System.out.println(String.format("Processed %d documents from the first input file", docNum));
        }
        
        oneRec = oneRec.trim() + System.getProperty("line.separator");
        
        ParsedQuestion q = null;
        try {
          q=YahooAnswersParser.parse(oneRec, false);
        } catch (Exception e) {
          // If <bestanswer>...</bestanswer> is missing we may end up here...
          // This is a bit funny, because this element is supposed to be mandatory,
          // but it's not.
          System.err.println("Skipping due to parsing error, exception: " + e);
          skipErrorQty++;
          continue;
        }
        if (hSubj.contains(q.mQuestion.trim())) {
          //System.out.println(String.format("Skipping uri='%s', question='%s'", q.mQuestUri, q.mQuestion));
          skipOverlapQty++;
          continue;
        }
        
        out.write(oneRec);
      }
      System.out.println(String.format("Processed %d documents, skipped because of overlap/errors %d/%d documents", 
                                        docNum - 1, skipOverlapQty, skipErrorQty));
      out.close();
    } catch (ParseException e) {
      Usage("Cannot parse arguments");
    } catch(Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
  }
}
