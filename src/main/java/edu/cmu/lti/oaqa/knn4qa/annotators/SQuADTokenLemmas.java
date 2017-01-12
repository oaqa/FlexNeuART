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
import org.apache.uima.cas.CASException;
import org.apache.uima.util.CasCopier;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;


import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.SQuADIntermCollectionReader;
import edu.cmu.lti.oaqa.knn4qa.types.*;

/**
 * The annotator that tokenizes and lemmatizes SQuAD questions and passages. 
 * 
 * @author Leonid Boytsov
 *
 */
public class SQuADTokenLemmas extends JCasAnnotator_ImplBase {
  private BasicEngine                   mTokenizerEngine;
  
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    mTokenizerEngine = new BasicEngine(aContext, false /* No POS tagging */);
  }

  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    // 1. Annotate the passage
    mTokenizerEngine.process(aJCas);
    // 2. Annotate all questions
    JCas questView = null;
    try {
      questView = aJCas.getView(SQuADIntermCollectionReader.QUESTION_VIEW);
    } catch (CASException e) {
      throw new AnalysisEngineProcessException(new Exception("No question view in the CAS!"));      
    }
    
    for (Question q : JCasUtil.select(questView, Question.class)) {
      JCas tmp = null;
      try {
        tmp = mTokenizerEngine.borrowJCas();
        tmp.setDocumentLanguage(SQuADIntermCollectionReader.DOCUMENT_LANGUAGE);
        tmp.setDocumentText(q.getCoveredText());
        mTokenizerEngine.process(tmp);
        CasCopier   copier = new CasCopier(tmp.getCas(), questView.getCas());
        for (Token tok : JCasUtil.select(tmp, Token.class)) {
          /*
           * This is a deep copy, however, it won't adjust positions of the referenced features
           * (in our case the only one of interest is Lemma) 
           */
          Token dstTok = (Token)copier.copyFs(tok);
          dstTok.setBegin(tok.getBegin() + q.getBegin());
          dstTok.setEnd(tok.getEnd() + q.getBegin());
          dstTok.addToIndexes();
        }
      } catch (Exception e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      } finally {
        if (tmp != null) mTokenizerEngine.returnJCas(tmp);
      }

    }
  }

}
