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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.*;

import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.XmlIterator;

/**
 * An application that splits a collection into several parts, e.g., the training, dev, and test sets. 
 * The user selects probabilities and parts' names. 
 * 
 * @author Leonid Boytsov
 *
 */
public class YahooAnswersCollectionSplitter extends CollectionSplitterBase {
  
  private static String DOCUMENT_TAG = "document";

  public static void main(String[] args) {
    mAppName = "Collection splitter for Yahoo Answers format";
        
    try {
      parseOptions(args);

      InputStream  input = CompressUtils.createInputStream(mInputFileName);
      
      XmlIterator  inpIter = new XmlIterator(input, DOCUMENT_TAG);
      
      String oneRec = inpIter.readNext();
      int docNum = 1;
      for (; !oneRec.isEmpty() ; ++docNum, oneRec = inpIter.readNext()) {
        double p = Math.random();
        
        if (docNum % 1000 == 0) {
          System.out.println(String.format("Processed %d documents", docNum));
        }
        
        BufferedWriter out = null;
        
        for (int partId = 0; partId < mPartNames.length; ++partId) {
          double pp = mProbs.get(partId);
          if (p <= pp || partId + 1 == mPartNames.length) {
            out = mOutFiles[partId];
            break;
          }
          p -= pp;
        }

        oneRec = oneRec.trim() + System.getProperty("line.separator");
        out.write(oneRec);
      }
      System.out.println(String.format("Processed %d documents", docNum - 1));
      // It's important to close all the streams here!
      closeFiles();
    } catch (ParseException e) {
      showUsage("Cannot parse arguments");
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
  }
}
