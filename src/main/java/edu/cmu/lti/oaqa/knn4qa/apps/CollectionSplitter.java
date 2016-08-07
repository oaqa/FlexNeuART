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
import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersReader;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;

/**
 * An application that splits a collection into several parts, e.g., the training, dev, and test sets. 
 * The user selects probabilities and parts' names. 
 * 
 * @author Leonid Boytsov
 *
 */
public class CollectionSplitter {
  static void Usage(String err) {
    System.err.println("Error: " + err);
    System.err.println("Usage: " 
                       + "-i <Input file> "
                       + "-o <Output file prefix> "
                       + "-p <Comma separated probabilities e.g., 0.1,0.2,0.7> " 
                       + "-n <Comma separated part names, e.g., dev,test,train> "
                       );
    System.exit(1);
  }

  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption("i", null, true, "Input file");
    options.addOption("o", null, true, "Output file prefix");
    options.addOption("p", null, true, "Comma separated probabilities e.g., 0.1,0.2,0.7.");
    options.addOption("n", null, true, "Comma separated part names, e.g., dev,test,train");

    CommandLineParser parser = new org.apache.commons.cli.GnuParser(); 
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      InputStream  input = null;
      
      if (cmd.hasOption("i")) {
        input = CompressUtils.createInputStream(cmd.getOptionValue("i"));
      } else {
        Usage("Specify Input file"); 
      }

      ArrayList<Double> probs = new ArrayList<Double>();
      String []         partNames = null;
      
      if (cmd.hasOption("p")) {
        String parts[] = cmd.getOptionValue("p").split(",");

        try {
          double sum = 0;
          for (String s: parts) {
            double p = Double.parseDouble(s);
            if (p <= 0 || p > 1) Usage("All probabilities must be in the range (0,1)");
            sum += p;
            probs.add(p);
          }
            
          if (Math.abs(sum - 1.0) > Float.MIN_NORMAL) {
            Usage("The sum of probabilities should be equal to 1, but it's: " + sum);
          }
        } catch (NumberFormatException e ) {
          Usage("Can't convert some of the probabilities to a floating-point number.");
        }
      } else {
        Usage("Specify part probabilities.");
      }
      
      if (cmd.hasOption("n")) {
        partNames = cmd.getOptionValue("n").split(",");
        
        if (partNames.length != probs.size())
          Usage("The number of probabilities is not equal to the number of parts!");
      } else {
        Usage("Specify part names");
      }
      
      BufferedWriter [] outFiles = new BufferedWriter[partNames.length];  
      
      if (cmd.hasOption("o")) {
        String outPrefix = cmd.getOptionValue("o");

        for (int partId = 0; partId < partNames.length; ++partId) {
          outFiles[partId] =  new BufferedWriter(
                                    new OutputStreamWriter(
                                         CompressUtils.createOutputStream(outPrefix+"_" + partNames[partId]+ ".gz")));
        }
      } else {
        Usage("Specify Output file prefix");      
      }
      
      System.out.println("Using probabilities:");
      for (int partId = 0; partId < partNames.length; ++partId) 
        System.out.println(partNames[partId] + " : " + probs.get(partId));
      System.out.println("=================================================");  
      
      XmlIterator   inpIter = new XmlIterator(input, YahooAnswersReader.DOCUMENT_TAG);
      
      String oneRec = inpIter.readNext();
      int docNum = 1;
      for (; !oneRec.isEmpty() ; ++docNum, oneRec = inpIter.readNext()) {
        double p = Math.random();
        
        if (docNum % 1000 == 0) {
          System.out.println(String.format("Processed %d documents", docNum));
        }
        
        BufferedWriter out = null;
        
        for (int partId = 0; partId < partNames.length; ++partId) {
          double pp = probs.get(partId);
          if (p <= pp || partId + 1 == partNames.length) {
            out = outFiles[partId];
            break;
          }
          p -= pp;
        }

        oneRec = oneRec.trim() + System.getProperty("line.separator");
        out.write(oneRec);
      }
      System.out.println(String.format("Processed %d documents", docNum - 1));
      // It's important to close all the streams here!
      for (BufferedWriter f : outFiles) f.close();
    } catch (ParseException e) {
      Usage("Cannot parse arguments");
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
  }
}
