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

import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.YahooAnswersReader;
import edu.cmu.lti.oaqa.knn4qa.qaintermform.*;
import edu.cmu.lti.oaqa.knn4qa.squad.*;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;


/**
 * An application that splits a Stanford SQuAD collection file into several parts, 
 * e.g., the training, translation and dev parts. 
 * The user selects probabilities and parts' names. 
 * 
 * @author Leonid Boytsov
 *
 */
public class SQuADCollectionSplitter extends CollectionSplitterBase {
  
  public static void main(String[] args) {
    mAppName = "Collection splitter for SQuAD format";
    
    try {
      parseOptions(args);
      
      SQuADReader  r = new SQuADReader(mInputFileName);
      
      int partQty = mPartNames.length;
      
      ArrayList<ArrayList<SQuADEntry>>   partParagrs = new ArrayList<ArrayList<SQuADEntry>>();
      
      for (int i = 0; i < partQty; ++i)
        partParagrs.add(new ArrayList<SQuADEntry>());
      
      QAData qd = new QAData(1);
      
      for (SQuADEntry e : r.mData.data) {
        // We will split at the paragraph level
        for (SQuADParagraph onePar : e.paragraphs) {
          double prob = Math.random();
          for (int partId = 0; partId < mPartNames.length; ++partId) {
            double pp = mProbs.get(partId);
            if (prob <= pp || partId + 1 == mPartNames.length) {
              partParagrs.get(partId).add(new SQuADEntry(e.title, onePar));
              break;
            }
            prob -= pp;
          }
        }
      }
      
      // Now let's write results
      for (int i = 0; i < partQty; ++i) {
        SQuADWriter.writeEntries(r.mData.version, partParagrs.get(i), mOutFiles[i]);
      }
    
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
