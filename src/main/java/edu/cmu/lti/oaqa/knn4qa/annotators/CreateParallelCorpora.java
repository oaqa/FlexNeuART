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
package edu.cmu.lti.oaqa.knn4qa.annotators;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import java.io.*;
import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.knn4qa.types.Answer;
import edu.cmu.lti.oaqa.knn4qa.types.Question;


/**
 * 
 * Creates parallel corpora for various features (text, bigrams, etc, ...); This class is thread-safe.
 * 
 * @author Leonid Boytsov
 *
 */

public class CreateParallelCorpora extends JCasAnnotator_ImplBase {
  protected static final String NL = System.getProperty("line.separator");
  private static final Logger logger = LoggerFactory.getLogger(CreateParallelCorpora.class);
  
  String mQuestionText;
  String mQuestionTextUnlemm;
  
  String mQuestURI;
  static private ExtractTextReps mTextRepExtract;
  static private ExtractTextReps mTextUnlemmRepExtract;
  
  static BufferedWriter mQuestionTextWriter;
  static BufferedWriter mAnswerTextWriter;
  static BufferedWriter mQuestionTextUnlemmWriter;
  static BufferedWriter mAnswerTextUnlemmWriter;  
  
  private static final String PARAM_STOPWORD_FILE = "StopWordFile";
  
  private static final String PARAM_QUESTION_PREFIX = "QuestionFilePrefix";
  private static final String PARAM_ANSWER_PREFIX   = "AnswerFilePrefix";
  
  private static final String PARAM_DUMP_TEXT         = "DumpText";
  private static final String PARAM_DUMP_TEXT_UNLEMM  = "DumpTextUnlemm";
  
  private static boolean mDumpText      = false;
  private static boolean mDumpTextUnlemm= false;

  private static int mIOState = 0;

  public static final boolean STRICTLY_GOOD_TOKENS_FOR_TRANSLATION = false;
  

  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
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
  }

  /**
   * @param stopWordFileName    the name of the stopword file
   * @throws Exception
   */
  static synchronized private void initTextExtract(String stopWordFileName)
      throws Exception {
    mTextRepExtract = new ExtractTextReps(stopWordFileName, true);
    mTextUnlemmRepExtract = new ExtractTextReps(stopWordFileName, false);
  }
  
  static synchronized private void initOutput(String questionPrefix, String answerPrefix) throws IOException {
    if (mIOState  != 0) return;
    
    if (mDumpText) {
      mQuestionTextWriter = new BufferedWriter(new FileWriter(questionPrefix + "_text"));
      mAnswerTextWriter   = new BufferedWriter(new FileWriter(answerPrefix   + "_text"));
    }
    
    if (mDumpTextUnlemm) {
      mQuestionTextUnlemmWriter = new BufferedWriter(new FileWriter(questionPrefix + "_text_unlemm"));
      mAnswerTextUnlemmWriter   = new BufferedWriter(new FileWriter(answerPrefix   + "_text_unlemm"));
    }    
   
    mIOState = 1;
  }

  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    Collection<Question> qs = JCasUtil.select(aJCas, Question.class);
    
    long threadId = Thread.currentThread().getId();
    logger.info(String.format("Process, thread # %d",threadId));
    
    if (!qs.isEmpty()) {
      if (qs.size() != 1) throw new AnalysisEngineProcessException(new Exception(
          String.format("Wrong # (%d) of question annotations, should be one", qs.size())));
      
      GoodTokens goodToks = mTextRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_TRANSLATION);
      GoodTokens goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_TRANSLATION);
      
      if (mDumpText)        mQuestionText   = mTextRepExtract.getText(goodToks);
      if (mDumpTextUnlemm)  mQuestionTextUnlemm=mTextUnlemmRepExtract.getText(goodToksUnlemm);
      
      mQuestURI = qs.iterator().next().getUri();
      
    } else {
      Collection<Answer> qa = JCasUtil.select(aJCas, Answer.class);
      // Yes, we can get an empty CAS, b/c we cannot drop existing one!
      if (qa.isEmpty()) return;
      if (qa.size() != 1) throw new AnalysisEngineProcessException(new Exception(
          String.format("Wrong # (%d) of answer annotations, should be one or zero", qa.size())));
      Answer an = qa.iterator().next();
      if (!an.getUri().equals(mQuestURI)) {
        throw new AnalysisEngineProcessException(new Exception(
          String.format("Bug, different URIs for question (%s) and answer (%s)", mQuestURI, an.getUri())));
      }
           
      try {
        //if (an.getIsBest())
        doOutput(aJCas,
                   mQuestionText,
                   mQuestionTextUnlemm);
      } catch (IOException e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }      
    }
    
  }

  static synchronized private void doOutput(JCas aJCas,
                                            String questionText,
                                            String questionTextUnlemm) throws IOException {
    GoodTokens goodToks = mTextRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_TRANSLATION);
    GoodTokens goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_TRANSLATION);
    
    if (mDumpText) {
      String answerText = mTextRepExtract.getText(goodToks);
      if (!questionText.isEmpty() && !answerText.isEmpty()) {
        mQuestionTextWriter.write(questionText + NL);
        mAnswerTextWriter.write(answerText + NL);
      }
    }
    if (mDumpTextUnlemm) {
      String answerTextUnlemm = mTextUnlemmRepExtract.getText(goodToksUnlemm);
      if (!questionTextUnlemm.isEmpty() && !answerTextUnlemm.isEmpty()) {
        mQuestionTextUnlemmWriter.write(questionTextUnlemm + NL);
        mAnswerTextUnlemmWriter.write(answerTextUnlemm + NL);
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

  static synchronized private void finishOutput() throws IOException {
    if (mIOState != 1) return;
    
    if (mDumpText) {
      mQuestionTextWriter.close();
      mAnswerTextWriter.close();
    }
    
    if (mDumpTextUnlemm) {
      mQuestionTextUnlemmWriter.close();
      mAnswerTextUnlemmWriter.close();
    }
    
    mIOState = 2;
  }
  

}
