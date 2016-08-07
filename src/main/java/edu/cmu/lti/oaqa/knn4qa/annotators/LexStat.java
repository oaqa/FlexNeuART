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

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A stupid sentence counter (shouldn't be run in a distributed system).
 */
public class LexStat extends JCasAnnotator_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(LexStat.class);
  
  private int mSentQty = 0; 
  private int mTokQty = 0;
  
  private synchronized void incSentQty() {
    ++mSentQty;
  }

  private synchronized void incTokQty() {
    ++mTokQty;
  }

  
  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    
    for (Sentence s : JCasUtil.select(jcas, Sentence.class)) {
      incSentQty();
    }
    for (Token s : JCasUtil.select(jcas, Token.class)) {
      incTokQty();
    }    
  }
  
  @Override
  public void collectionProcessComplete() {
    logger.info(String.format("# of sentences/tokens: %d/%d, thread %d", 
        mSentQty, mTokQty, Thread.currentThread().getId()));    
  }
}

