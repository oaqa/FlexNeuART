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
package edu.cmu.lti.oaqa.knn4qa.annotators;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.collection_reader.SQuADIntermCollectionReader;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.types.*;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;

public class SQuADCreateParallelCorpora extends JCasAnnotator_ImplBase {
  protected static final String NL = System.getProperty("line.separator");
  private static final Logger logger = LoggerFactory.getLogger(SQuADCreateParallelCorpora.class);
  
  String mQuestionText;
  String mQuestionTextUnlemm;
  
  String mQuestURI;
  static private SQuADExtractTextReps mTextRepExtract;
  static private SQuADExtractTextReps mTextUnlemmRepExtract;
  
  static BufferedWriter mQuestionTextWriter;
  static BufferedWriter mAnswerTextWriter;
  static BufferedWriter mQuestionTextUnlemmWriter;
  static BufferedWriter mAnswerTextUnlemmWriter;  
  
  private static final String PARAM_STOPWORD_FILE = "StopWordFile";
  
  private static final String PARAM_QUESTION_PREFIX = "QuestionFilePrefix";
  private static final String PARAM_ANSWER_PREFIX   = "AnswerFilePrefix";
  
  private static final String PARAM_DUMP_TEXT         = "DumpText";
  private static final String PARAM_DUMP_TEXT_UNLEMM  = "DumpTextUnlemm";
  
  private static boolean mDumpText       = false;
  private static boolean mDumpTextUnlemm = false;

  private static int mInitState = 0;

  public static final boolean STRICTLY_GOOD_TOKENS_FOR_TRANSLATION = false; 
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    synchronized(this.getClass()) {
      if (mInitState == 0) {
        
        super.initialize(aContext);
        
        String questionPrefix = (String)aContext.getConfigParameterValue(PARAM_QUESTION_PREFIX);
        String answerPrefix = (String)aContext.getConfigParameterValue(PARAM_ANSWER_PREFIX);
        
        Boolean tmpb = null;
        
        tmpb = (Boolean)aContext.getConfigParameterValue(PARAM_DUMP_TEXT);
        if (null != tmpb) mDumpText = tmpb;
        
        tmpb = (Boolean)aContext.getConfigParameterValue(PARAM_DUMP_TEXT_UNLEMM);
        if (null != tmpb) mDumpTextUnlemm = tmpb;
      
        try {
          initOutput(questionPrefix, answerPrefix);
          
          String tmps = (String)aContext.getConfigParameterValue(PARAM_STOPWORD_FILE);
          if (tmps == null) throw new ResourceInitializationException(
                                        new Exception("Missing parameter '" + PARAM_STOPWORD_FILE + "'"));
          initTextExtract(tmps);
        } catch (Exception e) {
          e.printStackTrace();
          throw new ResourceInitializationException(e);
        }
        long threadId = Thread.currentThread().getId();
        logger.info(String.format("Created an engine, thread # %d",threadId));
        
        mInitState = 1;
      }
    }
  }  
  
  /**
   * @param stopWordFileName    the name of the stopword file
   * @throws Exception
   */
  static synchronized private void initTextExtract(String stopWordFileName)
      throws Exception {
    mTextRepExtract = new SQuADExtractTextReps(stopWordFileName, true);
    mTextUnlemmRepExtract = new SQuADExtractTextReps(stopWordFileName, true);
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {      
    // Process the questions and write question-answer pairs
    JCas questView = null;
    try {
      questView = aJCas.getView(SQuADIntermCollectionReader.QUESTION_VIEW);
    } catch (CASException e) {
      throw new AnalysisEngineProcessException(new Exception("No question view in the CAS!"));      
    }
    
    for (FactoidQuestion coveringQuest : JCasUtil.select(questView, FactoidQuestion.class)) {
            
      GoodTokensSQuAD goodToksQuest = mTextRepExtract.getGoodTokens(questView, coveringQuest);
      String questionText = mTextRepExtract.getText(goodToksQuest); 
    
      GoodTokensSQuAD goodToksQuestUnlemm = mTextUnlemmRepExtract.getGoodTokens(questView, coveringQuest);
      String questionTextUnlemm = mTextRepExtract.getText(goodToksQuestUnlemm); 
      
      HashSet<Sentence> seenSentence = new HashSet<Sentence>();
      
      // We will create a QA pair for every sentence covering an answer
      int aQty = coveringQuest.getAnswers().size();
      
      for (int ia = 0; ia < aQty; ++ia) {   
        FactoidAnswer a = (FactoidAnswer) coveringQuest.getAnswers().get(ia);
                                           
        /*
         * This loop is not the most efficient way to check if an answer is covered by a sentence.
         * However, because the number of sentence is pretty limited, it should be fine.
         * In fact, extracting answer-type features is much more resource intensive.
         */
        for (Sentence answerCoveringSent : JCasUtil.select(aJCas, Sentence.class)) {
          if (seenSentence.contains(answerCoveringSent)) break;
          if (answerCoveringSent.getBegin() <= a.getBegin() && a.getEnd() <= answerCoveringSent.getEnd()) {
            seenSentence.add(answerCoveringSent);
            
       
            try {    
              if (mDumpText) {
                GoodTokensSQuAD goodToksAnsw = mTextRepExtract.getGoodTokens(aJCas, answerCoveringSent);  
                String answText = mTextRepExtract.getText(goodToksAnsw);
                
                mQuestionTextWriter.write(questionText + NL);
                mAnswerTextWriter.write(answText + NL);
              }
              
              if (mDumpTextUnlemm) {
                GoodTokensSQuAD goodToksAnswUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, answerCoveringSent);                 
                String answTextUnlemm = mTextUnlemmRepExtract.getText(goodToksAnswUnlemm);
                
                mQuestionTextUnlemmWriter.write(questionTextUnlemm + NL);
                mAnswerTextUnlemmWriter.write(answTextUnlemm + NL);
              }
             
            } catch (IOException e) {
              e.printStackTrace();
              logger.error("Error writing parallel corpus");
              throw new AnalysisEngineProcessException(e);
            }              
          }
        }
      }        
    }
  }
  
  @Override
  public void collectionProcessComplete() throws AnalysisEngineProcessException {
    long threadId = Thread.currentThread().getId();
    logger.info(String.format("collectionProcessComplete, thread # %d",threadId));    
    
    try {
      finishOutput();
    } catch (IOException e) {
      e.printStackTrace();
      throw new AnalysisEngineProcessException(e);
    }
  }
  
  static synchronized private void initOutput(String questionPrefix, String answerPrefix) throws IOException {
    if (mInitState  != 0) return;
    
    if (mDumpText) {
      mQuestionTextWriter = new BufferedWriter(new FileWriter(questionPrefix + "_text"));
      mAnswerTextWriter   = new BufferedWriter(new FileWriter(answerPrefix   + "_text"));
    }
    
    if (mDumpTextUnlemm) {
      mQuestionTextUnlemmWriter = new BufferedWriter(new FileWriter(questionPrefix + "_text_unlemm"));
      mAnswerTextUnlemmWriter   = new BufferedWriter(new FileWriter(answerPrefix   + "_text_unlemm"));
    }    
   
    mInitState = 1;
  }
  
  static synchronized private void finishOutput() throws IOException {
    if (mInitState != 1) return;

    if (mDumpText) {
      mQuestionTextWriter.close();
      mAnswerTextWriter.close();
    }
    if (mDumpTextUnlemm) {
      mQuestionTextUnlemmWriter.close();
      mAnswerTextUnlemmWriter.close();
    }
    
    mInitState = 2;
  }
}
