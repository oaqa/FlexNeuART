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

package edu.cmu.lti.oaqa.knn4qa.apps;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.google.gson.Gson;

import edu.cmu.lti.oaqa.knn4qa.qaintermform.*;
import edu.cmu.lti.oaqa.knn4qa.squad.*;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;

import java.io.*;
import java.util.HashMap;

/**
 * This class converts a split SQuAD collection to a bunch of files an intermediate JSON format.
 * <p>Note the following:</p>
 * <ul>
 * <li>The input collection is split into several parts that are named as specified by the class SQuADInputFiles.
 * <li>The split occurred at the paragraph/passage level, so we need a global numbering of passages that is the same for 
 *     all SQuAD parts. We also prepend each passage id with @-symbol to make it distinct from any id in the 
 *     non-SQuAD part of Wikipedia.
 * </ul>
 */

public class SQuAD2IntermConverter {
  private static final String INPUT_SQUAD_DIR = "input_squad_dir";
  
  private static Options    mOptions = new Options();
    
  private static Gson mGSON = new Gson();
  
  protected static final String NL = System.getProperty("line.separator");
  
  static void showUsage(String err) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(SQuAD2IntermConverter.class.getCanonicalName(), mOptions);      
    System.exit(1);
  }
  
  static void showUsageSpecify(String optName) {
    showUsage("Specify: '" + optName + "'");
  }
  
  public static void main(String[] args) {
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    
    mOptions.addOption(INPUT_SQUAD_DIR,     null, true,
        "Directory containing split SQuAD files as well as the dev1 (we expect certain hardcoded names)");
    
    try {
      CommandLine cmd = parser.parse(mOptions, args);
   
      String inputSQuADDir = cmd.getOptionValue(INPUT_SQUAD_DIR);
      if (inputSQuADDir == null) showUsageSpecify(INPUT_SQUAD_DIR);
                  
      int fileQty = SQuADInputFiles.mIntermInputFiles.length;
      
      //HashMap<String,Integer> hCurrPassageId = new HashMap<String,Integer>(); 
      
      int globalPassageId = 0, globalQuestionId = 0;
      
      for (int fileId = 0; fileId < fileQty; ++fileId) {
        
        String inputFileName = inputSQuADDir + "/" + SQuADInputFiles.mInputFiles[fileId];
        
        SQuADReader r = new SQuADReader(inputFileName);        
        
        String outFileName = inputSQuADDir + "/" + SQuADInputFiles.mIntermInputFiles[fileId];
        BufferedWriter outFile = 
            new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outFileName)));        
        
        System.out.println("Reading: '" + inputFileName + "' writing to '" + outFileName);

        QAData qd = new QAData(1);
        
        for (SQuADEntry e : r.mData.data) {
          for (SQuADParagraph passage : e.paragraphs) {
            String title = SQuADWikiTitlesReader.decodeTitle(e.title);
            /*
            Integer currPassId = hCurrPassageId.get(title);
            if (currPassId == null) {
              currPassId = 0;
              hCurrPassageId.put(title, 0);
            }
            */
            
            // this will be different from any id from the non-SQuAD Wikipedia part            
            String passageId = "@" + globalPassageId;
            globalPassageId++;
            
            int questQty = 0;
            
            if (passage.qas != null) questQty = passage.qas.length;
            
            QAPassage qp = new QAPassage(title, passageId, passage.context, questQty);
            qd.passages[0] = qp; // There is exactly one passage in this element
            
            if (questQty > 0) {
              for (int i = 0; i < passage.qas.length; ++i) {
                String questId = "" + globalQuestionId;
                ++globalQuestionId;
                SQuADQuestionAnswers currPass = passage.qas[i];
                QAQuestion qq = new QAQuestion(questId, currPass.question);
                QAAnswer[] qansw = new QAAnswer[0];
                if (currPass.answers != null) {
                   qansw = new QAAnswer[currPass.answers.length];
                   for (int aid = 0; aid < currPass.answers.length; ++aid) {
                     SQuADAnswer sqAnsw = currPass.answers[aid];
                     qansw[aid] = new QAAnswer(sqAnsw.answer_start, sqAnsw.answer_start + sqAnsw.text.length(), sqAnsw.text);
                   }
                } 
                qq.answers = qansw;
                qp.questions[i] = qq;
              }
            }
            
            String passageJSON = mGSON.toJson(qd, qd.getClass()).replace('\n', ' ');
            
            outFile.write(passageJSON);
            outFile.write(NL);
            /*
            ++currPassId;
            hCurrPassageId.replace(title, currPassId);
            */
          }
        }
               
        if (outFile != null) {
          try {
            outFile.close();
          } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
          }
        }      
      }

      
    } catch (ParseException e) {
      showUsage("Cannot parse arguments");
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }      
    
  }    
}
