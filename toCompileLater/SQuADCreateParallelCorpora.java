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
  String mQuestionQFeaturesAll;
  
  static private SQuADExtractTextReps mTextUnlemmRepExtract;
  static private SQuADExtractTextReps mTextRepExtract;

  static BufferedWriter mQuestTextWriter;
  static BufferedWriter mPassTextWriter;
  
  static BufferedWriter mQuestTextUnlemmWriter;
  static BufferedWriter mPassTextUnlemmWriter;

  private static int mInitState = 0;
  
  private static final String PARAM_STOPWORD_FILE = "StopWordFile";
  
  private static final String PARAM_QUESTION_PREFIX  = "QuestionFilePrefix";
  private static final String PARAM_PASSAGE_PREFIX   = "PassageFilePrefix";
  
  
  @ConfigurationParameter(name = PARAM_STOPWORD_FILE, mandatory = true)
  private String mStopWordFileName;
  @ConfigurationParameter(name = PARAM_QUESTION_PREFIX, mandatory = true) 
  private String mQuestionPrefix;
  @ConfigurationParameter(name = PARAM_PASSAGE_PREFIX, mandatory = true) 
  private String mPassagePrefix;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    synchronized(this.getClass()) {
      if (mInitState == 0) {
              
        try {
          String textField = FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_FIELD_ID];
          
          mQuestTextWriter = new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + textField));
          mPassTextWriter   = new BufferedWriter(new FileWriter(mPassagePrefix   + "_" + textField));
          
          String textUnlemmField = FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_UNLEMM_FIELD_ID];
          
          mQuestTextUnlemmWriter = new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + textUnlemmField));
          mPassTextUnlemmWriter   = new BufferedWriter(new FileWriter(mPassagePrefix   + "_" + textUnlemmField));
                      
          mTextRepExtract = new SQuADExtractTextReps(mStopWordFileName, true /* lemmatizing */);
          mTextUnlemmRepExtract = new SQuADExtractTextReps(mStopWordFileName, false);
          
        } catch (Exception e) {
          e.printStackTrace();
          throw new ResourceInitializationException(e);
        }
        mInitState = 1;
      }
    }
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
      
      goodToksQuest = mTextUnlemmRepExtract.getGoodTokens(questView, coveringQuest);
      
      String questionTextUnlemm =  mTextUnlemmRepExtract.getText(goodToksQuest);  
      
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
            
            String answText = mTextRepExtract.getText(
                mTextRepExtract.getGoodTokens(aJCas, answerCoveringSent));
            String answTextUnlemm = mTextUnlemmRepExtract.getText(
                mTextUnlemmRepExtract.getGoodTokens(aJCas, answerCoveringSent));
           
            try {
              
              mQuestTextWriter.write(questionText + NL);
              mPassTextWriter.write(answText + NL);
              
              mQuestTextUnlemmWriter.write(questionTextUnlemm + NL);
              mPassTextUnlemmWriter.write(answTextUnlemm + NL);
                           
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
  
  static synchronized private void finishOutput() throws IOException {
    if (mInitState != 1) return;

    mQuestTextWriter.close();
    mPassTextWriter.close();

    mQuestTextUnlemmWriter.close();
    mPassTextUnlemmWriter.close();
        
    mInitState = 2;
  }
}
