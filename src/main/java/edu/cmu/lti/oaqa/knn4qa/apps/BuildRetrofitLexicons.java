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

import java.io.BufferedWriter;
import java.io.FileWriter;

import org.apache.commons.cli.*;

import edu.cmu.lti.oaqa.knn4qa.giza.GizaOneWordTranRecs;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndexFilterAndRecoder;

public class BuildRetrofitLexicons {
  public static final String OUT_FILE_PARAM = "l";
  public static final String OUT_FILE_DESC  = "An output 'lexicon' style for the retrofitting software https://github.com/mfaruqui/retrofitting";
  public static final String MIN_PROB_PARAM = "min_prob";
  public static final String MIN_PROB_DESC  = "A minimum translation probability";
  public static final String FORMAT_PARAM = "f";
  private static final String ORIG_TYPE="orig", UNWEIGHTED_TYPE = "unweighted", WEIGHTED_TYPE = "weighted";
  public static final String FORMAT_DESC  = "Output format: " + ORIG_TYPE + "," + UNWEIGHTED_TYPE + "," + WEIGHTED_TYPE;
  
  enum FormatType {
    kOrig,
    kUnweighted,
    kWeighted
  };
  
  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("BuildRetrofitLexicons", options );      
    System.exit(1);
  }
  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.GIZA_ROOT_DIR_PARAM,  null, true,  CommonParams.GIZA_ROOT_DIR_DESC);
    options.addOption(CommonParams.GIZA_ITER_QTY_PARAM,  null, true,  CommonParams.GIZA_ITER_QTY_DESC);
    options.addOption(CommonParams.MEMINDEX_PARAM,       null, true,  CommonParams.MEMINDEX_DESC);
    options.addOption(OUT_FILE_PARAM,                    null, true,  OUT_FILE_DESC);
    options.addOption(MIN_PROB_PARAM,                    null, true,  MIN_PROB_DESC);
    options.addOption(FORMAT_PARAM,                      null, true,  FORMAT_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      String gizaRootDir = cmd.getOptionValue(CommonParams.GIZA_ROOT_DIR_PARAM); 
      int gizaIterQty = -1;
      
      if (cmd.hasOption(CommonParams.GIZA_ITER_QTY_PARAM)) {
        gizaIterQty = Integer.parseInt(cmd.getOptionValue(CommonParams.GIZA_ITER_QTY_PARAM));
      } else {
        Usage("Specify: " + CommonParams.GIZA_ITER_QTY_PARAM, options);
      }
      String outFileName = cmd.getOptionValue(OUT_FILE_PARAM);
      if (null == outFileName) {
        Usage("Specify: " + OUT_FILE_PARAM, options);
      }
      
      String indexDir = cmd.getOptionValue(CommonParams.MEMINDEX_PARAM);
      
      if (null == indexDir) {
        Usage("Specify: " + CommonParams.MEMINDEX_DESC, options);
      }
      
      FormatType outType = FormatType.kOrig;
      
      String outTypeStr = cmd.getOptionValue(FORMAT_PARAM);
      
      if (null != outTypeStr) {
        if (outTypeStr.equals(ORIG_TYPE)) {
          outType = FormatType.kOrig;
        } else if (outTypeStr.equals(WEIGHTED_TYPE)) {
          outType = FormatType.kWeighted;
        } else if (outTypeStr.equals(UNWEIGHTED_TYPE)) {
          outType = FormatType.kUnweighted;
        } else {
          Usage("Unknown format type: " + outTypeStr, options);
        }
      }
      
      float minProb = 0;
      
      if (cmd.hasOption(MIN_PROB_PARAM)) {
        minProb = Float.parseFloat(cmd.getOptionValue(MIN_PROB_PARAM));
      } else {
        Usage("Specify: " + MIN_PROB_PARAM, options);
      }
      
      System.out.println(String.format("Saving lexicon to '%s' (output format '%s'), keep only entries with translation probability >= %f", 
                                        outFileName, outType.toString(), minProb));


      // We use unlemmatized text here, because lemmatized dictionary is going to be mostly subset of the unlemmatized one.
      InMemForwardIndex textIndex = new InMemForwardIndex(FeatureExtractor.indexFileName(indexDir, 
                                                            FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_UNLEMM_FIELD_ID]));
      InMemForwardIndexFilterAndRecoder filterAndRecoder = new InMemForwardIndexFilterAndRecoder(textIndex);

      String prefix = gizaRootDir + "/" + FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_UNLEMM_FIELD_ID] + "/";
      GizaVocabularyReader answVoc  = new GizaVocabularyReader(prefix + "source.vcb", filterAndRecoder);
      GizaVocabularyReader questVoc = new GizaVocabularyReader(prefix + "target.vcb", filterAndRecoder);


      GizaTranTableReaderAndRecoder gizaTable = 
          new GizaTranTableReaderAndRecoder(false, // we don't need to flip the table for the purpose 
                                             prefix + "/output.t1." + gizaIterQty,
                                             filterAndRecoder,
                                             answVoc, questVoc,
                                             (float)FeatureExtractor.DEFAULT_PROB_SELF_TRAN, 
                                             minProb);
      BufferedWriter outFile = new BufferedWriter(new FileWriter(outFileName));
      
      for (int srcWordId = 0; srcWordId <= textIndex.getMaxWordId(); ++srcWordId) {
        GizaOneWordTranRecs tranRecs = gizaTable.getTranProbs(srcWordId);
        
        if (null != tranRecs) {
          String wordSrc = textIndex.getWord(srcWordId);
          StringBuffer sb = new StringBuffer();
          sb.append(wordSrc);
          
          for (int k = 0; k < tranRecs.mDstIds.length; ++k) {
            float prob = tranRecs.mProbs[k];
            if (prob >= minProb) {
              int dstWordId = tranRecs.mDstIds[k];

              if (dstWordId == srcWordId && outType != FormatType.kWeighted) continue; // Don't duplicate the word, unless it's probability weighted
              
              sb.append(' ');
              String dstWord = textIndex.getWord(dstWordId);
              if (null == dstWord) {
                throw new Exception("Bug or inconsistent data: Couldn't retriev a word for wordId = " + dstWordId);
              }
              if (dstWord.indexOf(':') >= 0)
                throw new Exception("Illegal dictionary word '" + dstWord + "' b/c it contains ':'");
              sb.append(dstWord);
              if (outType != FormatType.kOrig) {
                sb.append(':');
                sb.append(outType == FormatType.kWeighted ? prob : 1);
              }
            }
          }
                    
          outFile.write(sb.toString());
          outFile.newLine();
        }
      }
      
      outFile.close();
    } catch (ParseException e) {
      e.printStackTrace();
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
    System.out.println("Terminated successfully!");

  }
  
}
