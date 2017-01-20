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
  
  static private SQuADExtractTextReps mTextRepExtract;

  static BufferedWriter mQuestTextWriter;
  static BufferedWriter mPassTextWriter;
  
  static BufferedWriter mQuestQfeatOnlyWriter;
  static BufferedWriter mPassQfeatOnlyWriter;   

  static BufferedWriter mQuestTextQfeatWriter;
  static BufferedWriter mPassTextQfeatWriter;
  
  static BufferedWriter mQuestEphyraSpacyWriter;
  static BufferedWriter mPassEphyraSpacyWriter;   
  static BufferedWriter mQuestEphyraDbpediaWriter;
  static BufferedWriter mPassEphyraDbpediaWriter;   
  static BufferedWriter mQuestEphyraAllentWriter;
  static BufferedWriter mPassEphyraAllentWriter;   

  static BufferedWriter mQuestLexSpacyWriter;
  static BufferedWriter mPassLexSpacyWriter;   
  static BufferedWriter mQuestLexDbpediaWriter;
  static BufferedWriter mPassLexDbpediaWriter;   
  static BufferedWriter mQuestLexAllentWriter;
  static BufferedWriter mPassLexAllentWriter;   
  
  private static int mInitState = 0;
  
  private static final String PARAM_STOPWORD_FILE         = "StopWordFile";
  private static final String PARAM_FOCUSWORD_FILE        = "FreqFocusWordFile";
  
  private static final String PARAM_QUESTION_PREFIX  = "QuestionFilePrefix";
  private static final String PARAM_PASSAGE_PREFIX   = "PassageFilePrefix";
  
  
  @ConfigurationParameter(name = PARAM_STOPWORD_FILE, mandatory = true)
  private String mStopWordFileName;
  // Let this one be mandatory, there's no harm reading a small file in every SQuAD pipeline
  @ConfigurationParameter(name = PARAM_FOCUSWORD_FILE, mandatory = true) 
  private String mFocusWordFile;
  @ConfigurationParameter(name = PARAM_QUESTION_PREFIX, mandatory = true) 
  private String mQuestionPrefix;
  @ConfigurationParameter(name = PARAM_PASSAGE_PREFIX, mandatory = true) 
  private String mPassagePrefix;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    synchronized(this.getClass()) {
      if (mInitState == 0) {
        
        String textField = FeatureExtractor.mFieldNames[FeatureExtractor.TEXT_FIELD_ID];
        
        try {
          mQuestTextWriter = new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + textField));
          mPassTextWriter   = new BufferedWriter(new FileWriter(mPassagePrefix   + "_" + textField));
          
          mQuestQfeatOnlyWriter = new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.QFEAT_ONLY));
          mPassQfeatOnlyWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.QFEAT_ONLY));

          mQuestTextQfeatWriter = new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.TEXT_QFEAT));
          mPassTextQfeatWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.TEXT_QFEAT));
          
          mQuestEphyraSpacyWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.EPHYRA_SPACY));
          mPassEphyraSpacyWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.EPHYRA_SPACY));
          
          mQuestEphyraDbpediaWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.EPHYRA_DBPEDIA));
          mPassEphyraDbpediaWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.EPHYRA_DBPEDIA));

          mQuestEphyraAllentWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.EPHYRA_ALLENT));
          mPassEphyraAllentWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.EPHYRA_ALLENT));
          

          mQuestLexSpacyWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.LEXICAL_SPACY));
          mPassLexSpacyWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.LEXICAL_SPACY));
          
          mQuestLexDbpediaWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.LEXICAL_DBPEDIA));
          mPassLexDbpediaWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.LEXICAL_DBPEDIA));

          mQuestLexAllentWriter =  new BufferedWriter(new FileWriter(mQuestionPrefix + "_" + FeatureExtractor.LEXICAL_ALLENT));
          mPassLexAllentWriter = new BufferedWriter(new FileWriter(mPassagePrefix + "_" + FeatureExtractor.LEXICAL_ALLENT));
          
          
          mTextRepExtract = new SQuADExtractTextReps(mStopWordFileName, mFocusWordFile, true /* lowercasing */);
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
      
      HashSet<Sentence> seenSentence = new HashSet<Sentence>();
      
      // We will create a QA pair for every sentence covering an answer
      int aQty = coveringQuest.getAnswers().size();
      
      for (int ia = 0; ia < aQty; ++ia) {   
        FactoidAnswer a = (FactoidAnswer) coveringQuest.getAnswers().get(ia);
        
        
        // Lexical question features: w-word and focus word/phrase
        String qLexical = mTextRepExtract.getQuestionAnnot(questView, coveringQuest, 
                                              true /* w-word */, true /* focus word */, false /* Ephyra type */);
        // Ephyra question type
        String qType = mTextRepExtract.getQuestionAnnot(questView, coveringQuest, 
                                              false /* w-word */, false /* focus word */, true /* Ephyra type */);        
        
        /*
         * This loop is not the most efficient way to check if an answer is covered by a sentence.
         * However, because the number of sentence is pretty limited, it should be fine.
         * In fact, extracting answer-type features is much more resource intensive.
         */
        for (Sentence answerCoveringSent : JCasUtil.select(aJCas, Sentence.class)) {
          if (seenSentence.contains(answerCoveringSent)) break;
          if (answerCoveringSent.getBegin() <= a.getBegin() && a.getEnd() <= answerCoveringSent.getEnd()) {
            seenSentence.add(answerCoveringSent);
            
            GoodTokensSQuAD goodToksAnsw = mTextRepExtract.getGoodTokens(aJCas, answerCoveringSent);      

            String answText       = mTextRepExtract.getText(goodToksAnsw);
            String answAllNER     = mTextRepExtract.getNER(aJCas, answerCoveringSent, true, true);
            String answDbpediaNER = mTextRepExtract.getNER(aJCas, answerCoveringSent, true, false);
            String answSpacyNER   = mTextRepExtract.getNER(aJCas, answerCoveringSent, false, true);
            
            try {                
              mQuestTextWriter.write(questionText + NL);
              mPassTextWriter.write(answText + NL);

              mQuestTextQfeatWriter.write(qLexical + " " + questionText + " " + qType + NL);
              mPassTextQfeatWriter.write(answText + " " + answAllNER + NL);
              
              mQuestQfeatOnlyWriter.write(qLexical + " " + qType + NL);
              mPassQfeatOnlyWriter.write(answAllNER + NL);
              
              mQuestEphyraAllentWriter.write(qType + NL);
              mPassEphyraAllentWriter.write(answAllNER + NL);
              
              mQuestEphyraDbpediaWriter.write(qType + NL);
              mPassEphyraDbpediaWriter.write(answDbpediaNER + NL);              
              
              mQuestEphyraSpacyWriter.write(qType + NL);
              mPassEphyraSpacyWriter.write(answSpacyNER + NL);

              mQuestLexAllentWriter.write(qLexical + NL);
              mPassLexAllentWriter.write(answAllNER + NL);
              
              mQuestLexDbpediaWriter.write(qLexical + NL);
              mPassLexDbpediaWriter.write(answDbpediaNER + NL);              
              
              mQuestLexSpacyWriter.write(qLexical + NL);
              mPassLexSpacyWriter.write(answSpacyNER + NL);              
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

    mQuestTextQfeatWriter.close();
    mPassTextQfeatWriter.close();

    mQuestQfeatOnlyWriter.close();
    mPassQfeatOnlyWriter.close();

    mQuestEphyraAllentWriter.close();
    mPassEphyraAllentWriter.close();

    mQuestEphyraDbpediaWriter.close();
    mPassEphyraDbpediaWriter.close();              

    mQuestEphyraSpacyWriter.close();
    mPassEphyraSpacyWriter.close();

    mQuestLexAllentWriter.close();
    mPassLexAllentWriter.close();
    
    mQuestLexDbpediaWriter.close();
    mPassLexDbpediaWriter.close();              
    
    mQuestLexSpacyWriter.close();
    mPassLexSpacyWriter.close();     
    
    mInitState = 2;
  }
}
