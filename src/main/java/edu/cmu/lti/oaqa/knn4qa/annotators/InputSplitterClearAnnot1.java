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

import java.util.ArrayList;

import org.apache.uima.UimaContext;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.CasCopier;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.AbstractCas;
import org.apache.uima.fit.component.JCasMultiplier_ImplBase;
import org.apache.uima.fit.util.JCasUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import edu.cmu.lti.oaqa.knn4qa.types.*; 

/**
 * 
 * Splits a CAS containing both question and answers into multiple
 * CASes (one per each answer and/or question). In addition,
 * it deletes low-quality answers and/or questions. To be able
 * to drop CASes one needs to configure the flow correctly. See,
 * the page here for details:
 * https://uima.apache.org/d/uimaj-2.7.0/tutorials_and_users_guides.html#ugr.tug.fc.using_fc_with_cas_multipliers
 * 
 * @author Leonid Boytsov
 *
 */

public class InputSplitterClearAnnot1 extends JCasMultiplier_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(InputSplitterClearAnnot1.class);
  
  private static final String PARAM_MIN_TOK_QTY   = "MinTokQty";
  private static final String PARAM_CHECK_QUALITY = "CheckQuality";
  private static final String PARAM_SKIP_ANSWERS  = "SkipAnswers";
  private static final String PARAM_INCLUDE_NOT_BEST = "IncludeNotBest";
  private static final String PARAM_DO_POS_TAGGING    = "DoPOSTagging";
  
  private int       mMinTokQty    = 0;
  private boolean   mCheckQuality = false;
  private boolean   mSkipAnswers  = false;
  private boolean   mIncludeNotBest = false;
  private boolean   mDoPOSTagging   = false;
  
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    Integer tmpInt = (Integer) aContext.getConfigParameterValue(PARAM_MIN_TOK_QTY);
    if (null != tmpInt) mMinTokQty = tmpInt;
    Boolean tmpBool = (Boolean) aContext.getConfigParameterValue(PARAM_CHECK_QUALITY);
    if (tmpBool != null) mCheckQuality = tmpBool;
    tmpBool = (Boolean) aContext.getConfigParameterValue(PARAM_SKIP_ANSWERS);
    if (tmpBool != null) mSkipAnswers = tmpBool;    
    tmpBool = (Boolean) aContext.getConfigParameterValue(PARAM_INCLUDE_NOT_BEST);
    if (tmpBool != null) mIncludeNotBest = tmpBool;
    tmpBool = (Boolean) aContext.getConfigParameterValue(PARAM_DO_POS_TAGGING);
    if (tmpBool != null) mDoPOSTagging = tmpBool;
    
    // quality checker needs POS tags
    if (mCheckQuality) mDoPOSTagging = true;

    logger.info(String.format("DoPOSTagging=%b MinTokQty=%d CheckQuality=%b SkipAnswers=%b IncludeNotBest=%b"
        , mDoPOSTagging, mMinTokQty, mCheckQuality, mSkipAnswers, mIncludeNotBest));
    
    mClearNLPEngine = new BasicEngine(aContext, mDoPOSTagging);
    mJCasFactory = mClearNLPEngine.createJCasFactory();
  }
  
  JCas                          mQuestJCas = null;
  ArrayList<JCas>               mAnswerJCas = new ArrayList<JCas>();
  ArrayList<Answer>             mAnswerAnnot = new ArrayList<Answer>();

  JCas                          mBaseJcas;

  boolean                       mGood = false;
  boolean                       mReleasedEmpty = false;
  private int                   mAnswId = -1;

  BasicEngine                   mClearNLPEngine;
  private JCasFactory           mJCasFactory;
 
  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    mBaseJcas = jcas;
    mGood = false;    
    mReleasedEmpty = false;
    
    //logger.info("*** Process! ***");
    
    releaseAll();
    
    Question yaq = null;
    
    {
      try {
        mQuestJCas = mJCasFactory.borrowJCas();
      } catch (ResourceInitializationException e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }
      
      mQuestJCas.setDocumentLanguage(mBaseJcas.getDocumentLanguage());        
      yaq = JCasUtil.selectSingle(mBaseJcas, Question.class);
      mQuestJCas.setDocumentText(yaq.getCoveredText());
      
      mClearNLPEngine.process(mQuestJCas);
      
      if (mCheckQuality && !SimpleTextQualityChecker.checkCAS(mQuestJCas, mMinTokQty)) {
        logger.info(String.format("Dropping question %s, text '%s'", 
                              yaq.getUri(), yaq.getCoveredText()));          
        return; // The question CAS will be released by the function releaseAll(),
                // which is invoked every time the function process(JCas jcas) is executed.
      }
    }
    
    if (!mSkipAnswers) {
     for (Answer yan : JCasUtil.select(jcas, Answer.class)) 
     if (yan.getIsBest() || mIncludeNotBest) {
        JCas answJCas = null;
        try {
          /*
           *  By default, a CAS Multiplier is only allowed to hold one output CAS instance at a time.
           *  However, we might need to have multiple output CASes.
           *  Using a separate analysis engine (inside BasicEngine) provides a work around.
           */
          answJCas = mJCasFactory.borrowJCas();
        } catch (Exception e) {
          e.printStackTrace();
          throw new AnalysisEngineProcessException(e);
        }
  
        answJCas.setDocumentText(yan.getCoveredText());
        answJCas.setDocumentLanguage(mBaseJcas.getDocumentLanguage());
        
        mClearNLPEngine.process(answJCas);
  
        try {
          if (mCheckQuality && !SimpleTextQualityChecker.checkCAS(answJCas, mMinTokQty)) {
            logger.info(String.format("Dropping answer %s, text '%s'", 
                yan.getId(), yan.getCoveredText()));
            continue;
          }
        } catch (Exception e) {
          e.printStackTrace();
          throw new AnalysisEngineProcessException(e); 
        }
        
        mAnswerJCas.add(answJCas); // will make it possible to release in the case of failure
        mAnswerAnnot.add(yan);
      }
      
      if (mAnswerJCas.isEmpty()) {
        logger.info(String.format("Dropping question %s, because there are no good answers.", 
            yaq.getUri(), yaq.getCoveredText()));      
        return; // No answer passed the check
      }
    }

    mAnswId = -1;    
    mGood  = true;
  }
  
  private void releaseAll() {
    if (mQuestJCas != null) {
      mJCasFactory.returnJCas(mQuestJCas);
      mQuestJCas = null;
    }
    for (JCas jcas: mAnswerJCas) {
      mJCasFactory.returnJCas(jcas);
    }
    mAnswerJCas.clear();
    mAnswerAnnot.clear();
  }
  public boolean hasNext() throws AnalysisEngineProcessException {
    return (mGood && (mAnswId < mAnswerJCas.size()))  ||
           (!mGood && !mReleasedEmpty);
  }

  public AbstractCas next() throws AnalysisEngineProcessException {
    JCas jCasDst = getEmptyJCas();
    
    jCasDst.setDocumentLanguage(mBaseJcas.getDocumentLanguage());
    
    /**
     * Problem is that old CAS seems to refuse to be dropped if 
     * the new CAS is not produced.
     */
    if (!mGood) {
      mReleasedEmpty = true;
      jCasDst.setDocumentText("");
      return jCasDst;
    }
    
    //logger.info("*** NEXT! ***");

    CasCopier   copier = new CasCopier(mBaseJcas.getCas(), jCasDst.getCas());                  
    
    if (mAnswId < 0) {
      Question yaq = JCasUtil.selectSingle(mBaseJcas, Question.class);
    
      jCasDst.setDocumentText(mQuestJCas.getDocumentText());
      // Copy tags produced by the sentence splitter, tagger, and tokenizer
      copyAnnotations(mQuestJCas, jCasDst);
      
      // After copying attributes, correct start/end 
      Question dstQuest = (Question)copier.copyFs(yaq);
      dstQuest.setBegin(0);
      dstQuest.setEnd(yaq.getCoveredText().length());
      // start/end are corrected, can now index 
      dstQuest.addToIndexes();                 
      
      mAnswId = 0;
      
      PrintInfoHelper.printInfo1(logger, jCasDst);
      
      return jCasDst;
    }
    
    JCas    answJCas = mAnswerJCas.get(mAnswId);
    Answer  yan      = mAnswerAnnot.get(mAnswId);
    
    jCasDst.setDocumentText(answJCas.getDocumentText());
    
    // After copying attributes, correct start/end indices
    Answer dstAnsw = (Answer)copier.copyFs(yan);
    dstAnsw.setBegin(0);
    dstAnsw.setEnd(yan.getCoveredText().length());
    // start/end are corrected, can now index 
    dstAnsw.addToIndexes();
    
    // Copy tags produced by the sentence splitter, tagger, and tokenizer
    copyAnnotations(answJCas, jCasDst);
    
    ++mAnswId; 
    
    PrintInfoHelper.printInfo1(logger, jCasDst);
    
    return jCasDst;
  }
  
  private void copyAnnotations(JCas jCasSrc, JCas jCasDst) {
    CasCopier   copier = new CasCopier(jCasSrc.getCas(), jCasDst.getCas());

    for (Sentence anSrc : JCasUtil.select(jCasSrc, Sentence.class)) {
      Sentence anDst = (Sentence) copier.copyFs(anSrc);
      anDst.addToIndexes();
    }
    
    for (Token anSrc : JCasUtil.select(jCasSrc, Token.class)) {
      Token anDst = (Token) copier.copyFs(anSrc);
      anDst.addToIndexes();      
    }    
    
    for (POS anSrc : JCasUtil.select(jCasSrc, POS.class)) {
      POS anDst = (POS) copier.copyFs(anSrc);
      anDst.addToIndexes();      
    }    
  }
}
