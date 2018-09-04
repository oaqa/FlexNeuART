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

import net.openhft.koloboke.collect.map.hash.HashIntObjMap;
import no.uib.cipr.matrix.sparse.SparseVector;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.cmu.lti.oaqa.knn4qa.embed.SparseEmbeddingReaderAndRecorder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.FrequentIndexWordFilterAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndexFilterAndRecoder;
import edu.cmu.lti.oaqa.knn4qa.simil.BM25SimilarityLucene;
import edu.cmu.lti.oaqa.knn4qa.utils.VocabularyFilterAndRecoder;

public class GenTranEmbeddings {

  static void Usage(String err, Options options) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "GenTranEmbeddings", options );      
    System.exit(1);
  }  

  public static final String OUT_FILE_PARAM = "o";
  public static final String OUT_FILE_DESC  = "An output file prefix";
  public static final String MAX_MODEL_ORDER_PARAM = "m";
  public static final String MAX_MODEL_ORDER_DESC  = "A maximum model order";
  public static final String MIN_PROB_PARAM = "p";
  public static final String MIN_PROB_DESC  = "A minimum probability";
  public static final String MAX_DIGIT_PARAM= "max_digit";
  public static final String MAX_DIGIT_DESC = "the maximum # of digits to print";

  

  
  public static void main(String[] args) {
    Options options = new Options();
    
    options.addOption(CommonParams.MEMINDEX_PARAM,            null, true,  CommonParams.MEMINDEX_DESC);
    options.addOption(CommonParams.GIZA_ROOT_DIR_PARAM,       null, true,  CommonParams.GIZA_ROOT_DIR_DESC);
    options.addOption(CommonParams.GIZA_ITER_QTY_PARAM,       null, true,  CommonParams.GIZA_ITER_QTY_DESC);
    options.addOption(OUT_FILE_PARAM,                         null, true,  OUT_FILE_DESC);
    options.addOption(MAX_MODEL_ORDER_PARAM,                  null, true,  MAX_MODEL_ORDER_DESC);
    options.addOption(MIN_PROB_PARAM,                         null, true,  MIN_PROB_DESC);
    options.addOption(MAX_DIGIT_PARAM,                        null, true,  MAX_DIGIT_DESC);
    options.addOption(CommonParams.MAX_WORD_QTY_PARAM,        null, true, CommonParams.MAX_WORD_QTY_PARAM);

    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      int maxWordQty = Integer.MAX_VALUE;
      
      String tmpi = cmd.getOptionValue(CommonParams.MAX_WORD_QTY_PARAM);
      
      if (null != tmpi) {
        maxWordQty = Integer.parseInt(tmpi);
      }      
      
      String memIndexPref = cmd.getOptionValue(CommonParams.MEMINDEX_PARAM);
      
      if (null == memIndexPref) {
        Usage("Specify '" + CommonParams.MEMINDEX_DESC + "'", options);
      } 
      String gizaRootDir = cmd.getOptionValue(CommonParams.GIZA_ROOT_DIR_PARAM);
      if (null == gizaRootDir) {
        Usage("Specify '" + CommonParams.GIZA_ROOT_DIR_PARAM + "'", options);
      }
      int gizaIterQty = -1;
      if (cmd.hasOption(CommonParams.GIZA_ITER_QTY_PARAM)) {
        gizaIterQty = Integer.parseInt(cmd.getOptionValue(CommonParams.GIZA_ITER_QTY_PARAM));
      }  
      if (gizaIterQty <= 0) {
        Usage("Specify '" + CommonParams.GIZA_ITER_QTY_DESC + "'", options);
      }
      int maxModelOrder = -1;
      if (cmd.hasOption(MAX_MODEL_ORDER_PARAM)) {
        maxModelOrder = Integer.parseInt(cmd.getOptionValue(MAX_MODEL_ORDER_PARAM));
      }
      String outFilePrefix = cmd.getOptionValue(OUT_FILE_PARAM);
      if (null == outFilePrefix) {
        Usage("Specify '" + OUT_FILE_DESC + "'", options);
      }
      
      float minProb = 0;
      
      if (cmd.hasOption(MIN_PROB_PARAM)) {
        minProb = Float.parseFloat(cmd.getOptionValue(MIN_PROB_PARAM));
      } else {
        Usage("Specify '" + MIN_PROB_DESC + "'", options);
      }
      
      int maxDigit = 5;
      if (cmd.hasOption(MAX_DIGIT_PARAM)) {
        maxDigit = Integer.parseInt(cmd.getOptionValue(MAX_DIGIT_PARAM));
      }
      
      // We use unlemmatized text here, because lemmatized dictionary is going to be mostly subset of the unlemmatized one.
      int fieldId = FeatureExtractor.TEXT_UNLEMM_FIELD_ID;
      
      String memFwdIndxName = FeatureExtractor.indexFileName(memIndexPref, FeatureExtractor.mFieldNames[fieldId]);
      
      FrequentIndexWordFilterAndRecoder filterAndRecoder = new FrequentIndexWordFilterAndRecoder(memFwdIndxName, maxWordQty);
      
      InMemForwardIndex    index = new InMemForwardIndex(memFwdIndxName);
      BM25SimilarityLucene simil = new BM25SimilarityLucene(FeatureExtractor.BM25_K1, FeatureExtractor.BM25_B, index);
      String prefix = gizaRootDir + "/" + FeatureExtractor.mFieldNames[fieldId] + "/";

      GizaVocabularyReader answVoc  = new GizaVocabularyReader(prefix + "source.vcb", filterAndRecoder);
      GizaVocabularyReader questVoc = new GizaVocabularyReader(prefix + "target.vcb", filterAndRecoder);
      
      GizaTranTableReaderAndRecoder answToQuestTran = new GizaTranTableReaderAndRecoder(false /* don't flip a translation table */,
                                                        prefix + "/output.t1." + gizaIterQty,
                                                        filterAndRecoder,
                                                        answVoc, questVoc,
                                                        (float)FeatureExtractor.DEFAULT_PROB_SELF_TRAN, 
                                                        minProb);
      
      int order = 0;
   
      System.out.println("Starting to compute the 0-order model");
      HashIntObjMap<SparseVector> currModel = SparseEmbeddingReaderAndRecorder.createTranVecDict(index, filterAndRecoder, minProb, answToQuestTran);
      System.out.println("0-order model is computed");      
      SparseEmbeddingReaderAndRecorder.saveDict(index, outFilePrefix + ".0", currModel, maxDigit);
      System.out.println("0-order model is saved");
      
      while (order < maxModelOrder) {
        ++order;
        System.out.println("Starting to compute the "+order+"-order model");
        currModel = SparseEmbeddingReaderAndRecorder.nextOrderDict(currModel, index, minProb, answToQuestTran);
        System.out.println(order+"-order model is computed");  
        SparseEmbeddingReaderAndRecorder.saveDict(index, outFilePrefix + "." + order, currModel, maxDigit);
        System.out.println(order+"-order model is saved");
      }

    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
    System.out.println("Terminated successfully!");

    
  }
}
