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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.google.common.base.Joiner;

import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.ParamHelper;

class ParsedLine {
  int mDocId = -1;
  int mWordId = -1;
  int mInDocQty = -1;
  public ParsedLine(int mDocId, int mWordId, int mInDocQty) {
    this.mDocId = mDocId;
    this.mWordId = mWordId;
    this.mInDocQty = mInDocQty;
  }
};

/**
 * An application that can convert a sparse bag-of-words UCI data set to
 * a knn4qa forward file. 
 * 
 * <p>It accepts two files: docword.<data set name>.txt.gz
 * and vocab.<data set name>.txt It doesn't use the class
 * that manipulates the forward file and produces the forward file 
 * in the format used at the moment of writing this utility.</p>  
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertUCIBagOfWords {

  private static final int REPORT_QTY = 1000000;
  private static final String INPUT_DIR_PARAM = "input_dir";
  private static final String INPUT_DIR_DESC = "Directory containing input data set";
  private static final String DATASET_NAME_PARAM = "dataset_name";
  private static final String DATASET_NAME_DESC = "Data set name";
  private static final String OUTPUT_FILE_PARAM = "out_file";
  private static final String OUTPUT_FILE_DESC = "Output file";

  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    if (options != null) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("ConvertUCIBagOfWords", options );
    }
    System.exit(1);
  }  
  
  static void UsageSpecify(String param, Options options) {
    Usage("Specify '" + param + "'", options);
  }      
  
  public static void main(String[] args) {        
    String optKeys[] = {
        INPUT_DIR_PARAM,
        DATASET_NAME_PARAM,
        OUTPUT_FILE_PARAM
    };
    String optDescs[] = {
        INPUT_DIR_DESC,
        DATASET_NAME_DESC,
        OUTPUT_FILE_DESC
    };
    
    boolean hasArg[] = {
        true,
        true,
        true,
    };
    
    ParamHelper prmHlp = null;
    
    try {

      prmHlp = new ParamHelper(args, optKeys, optDescs, hasArg);    
   
      CommandLine cmd = prmHlp.getCommandLine();
      Options     opt = prmHlp.getOptions();
      
      String inputDir = cmd.getOptionValue(INPUT_DIR_PARAM);
      if (null == inputDir) {
        UsageSpecify(INPUT_DIR_PARAM, opt);
      }
      String dataSetName = cmd.getOptionValue(DATASET_NAME_PARAM);
      if (null == dataSetName) {
        UsageSpecify(DATASET_NAME_PARAM, opt);
      }
      String outFileName = cmd.getOptionValue(OUTPUT_FILE_PARAM);
      if (null == outFileName) {
        UsageSpecify(OUTPUT_FILE_PARAM, opt);
      }
      readDict(inputDir, dataSetName);
    
      String fn = inputDir + "/docword." + dataSetName + ".txt.gz";
      
      BufferedWriter outFile = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outFileName)));
      
      long totalQty = 0;
      
      for (int pass = 0; pass < 2; ++pass) {
        BufferedReader r = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(fn)));
        
        String sDocQty = r.readLine();
        if (sDocQty == null) {
          throw new Exception("Cannot read the number of documents in line 1");
        }        
        String sWordQty = r.readLine();
        if (sWordQty == null) {
          throw new Exception("Cannot read the number of unique words in line 2");
        }
        String sDocLineQty = r.readLine();
        if (sDocLineQty == null) {
          throw new Exception("Cannot read the total number of word occurrences in line 3");
        }
        int  docQty = Integer.parseInt(sDocQty);
        int  wordQty = Integer.parseInt(sWordQty);
        long docLineQty = Long.parseLong(sDocLineQty);
        int ln = 3;
        String line = null; 
        
        System.out.println("The dictionary is read!");
        
        if (pass == 0) {
          
          long procDocLineQty = 0;
          while ((line = r.readLine()) != null) {
            ln++;
            procDocLineQty++;
            if (ln % REPORT_QTY == 0) 
              System.out.println("The first pass: " + ln + " lines of '" + fn + "' are processed");
            line = line.trim();
            if (line.isEmpty()) continue;
            ParsedLine lineData = parseLine(line, fn, ln);
            checkData(fn, docQty, wordQty, ln, lineData);
            int wordIdIndex = lineData.mWordId - 1;
            mWordQty.set(wordIdIndex, mWordQty.get(wordIdIndex) + 1);
            totalQty += lineData.mInDocQty;
          }
          System.out.println("The first pass finished: " + ln + " lines of '" + fn + "' are processed");
          if (procDocLineQty != docLineQty)
            throw new Exception("Data set may be corrputed, the total number of word entries declared in the header (" +
                                docLineQty + ") != the sum of in-doc qtys (" + procDocLineQty + ")");
          
        }
        if (pass == 1) {
          // First dump the vocabulary
          outFile.write(docQty + " " + totalQty);
          outFile.write("\n\n");
          
          for (int i = 0; i < mWordQty.size(); ++i) {
            outFile.write(mWordList.get(i) + "\t" + (i+1) + ":" + mWordQty.get(i) + "\n");
          }
          outFile.write("\n");
          System.out.println("The dictionary is written!");
          long procDocLineQty = 0;
         
          ArrayList<ParsedLine> currBuff = new ArrayList<ParsedLine>();
          
          while ((line = r.readLine()) != null) {
            ln++;
            procDocLineQty++;
            if (ln % REPORT_QTY == 0) 
              System.out.println("The second pass: " + ln + " lines of '" + fn + "' are processed");
            line = line.trim();
            if (line.isEmpty()) continue;
            ParsedLine lineData = parseLine(line, fn, ln);
            // A bit paranoid, but let's check data validity again
            checkData(fn, docQty, wordQty, ln, lineData);
            if (!currBuff.isEmpty() && currBuff.get(currBuff.size()-1).mDocId != lineData.mDocId) {
              outputOneDoc(currBuff, outFile);
              currBuff.clear();
            }
            currBuff.add(lineData);
          }
          outputOneDoc(currBuff, outFile);
          outFile.write("\n");
          outFile.close();
          // A bit paranoid, but let's check data validity again
          System.out.println("The second pass finished: " + ln + " lines of '" + fn + "' are processed");
          if (procDocLineQty != docLineQty)
            throw new Exception("Data set may be corrputed, the total number of word entries declared in the header (" +
                                docLineQty + ") != the sum of in-doc qtys (" + procDocLineQty + ")");
        }
        r.close();
      }
    } catch (ParseException e) {
      Usage("Cannot parse arguments: " + e, prmHlp != null ? prmHlp.getOptions() : null);
      e.printStackTrace();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }        
  }

  private static Joiner tabJoin  = Joiner.on('\t');
  
  private static void outputOneDoc(ArrayList<ParsedLine> currBuff, BufferedWriter outFile) throws IOException {
    if (currBuff.isEmpty()) return;
    int totDocQty = 0;
    
    ArrayList<String> str = new ArrayList<String>();
    for (ParsedLine e : currBuff) {
      totDocQty += e.mInDocQty;
      str.add(e.mWordId + ":" + e.mInDocQty);
    }
    outFile.write(currBuff.get(0).mDocId + "\n");
    str.add("\n");
    outFile.write(tabJoin.join(str));
    outFile.write("@ " + totDocQty +"\n");
  }

  private static void checkData(String fn, int docQty, int wordQty, int ln, ParsedLine lineData) throws Exception {
    if (lineData.mWordId < 1 || lineData.mWordId > wordQty) {
      throw new Exception("Wrong format of the file: '" + fn + "' line: " + ln + " wordId is not in the header-defined range");
    }            
    if (lineData.mDocId < 1 || lineData.mDocId > docQty) {
      throw new Exception("Wrong format of the file: '" + fn + "' line: " + ln + " docId is not in the header-defined range");
    }
    if (lineData.mInDocQty < 1) {
      throw new Exception("Wrong format of the file: '" + fn + "' line: " + ln + " inDocQty < 1");
    }
  }

  private static ParsedLine parseLine(String line, String fileName, int ln) throws Exception {
    String parts[] = line.split(" ");
    if (parts.length != 3) {
      throw new Exception("Wrong format of the file: '" + fileName + "' expecting three fields in line: " + ln);
    }
    
    Integer docId  = Integer.parseInt(parts[0]);
    Integer wordId = Integer.parseInt(parts[1]);
    Integer inDocQty = Integer.parseInt(parts[2]);
    return new ParsedLine(docId, wordId, inDocQty);
  }

  private static void readDict(String inputDir, String dataSetName) throws IOException {
    String fn = inputDir + "/vocab." + dataSetName + ".txt";
    
    BufferedReader r = new BufferedReader(new FileReader(new File(fn)));
   
    String line;
    
    while ((line=r.readLine())!=null) {
      String word = line.trim();
      mWordList.add(word);
      mWordQty.add(0);
    }
    r.close();
  }

  private static ArrayList<String> mWordList = new ArrayList<String>();
  private static ArrayList<Integer> mWordQty = new ArrayList<Integer>(); 
}
