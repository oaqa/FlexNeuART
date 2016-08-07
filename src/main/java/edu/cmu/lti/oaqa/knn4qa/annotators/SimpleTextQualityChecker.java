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

import java.util.Collection;

import org.apache.uima.jcas.JCas;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;

/**
 *  A quality checker class implementing a simple filter described
 *  on p. 363 (step 2) of the paper:
 *  Surdeanu et al. 2011 Learning to Rank Answers to Non-Factoid Questions from Web Collections. 
 *  
 *  <p>This class has a performance overhead, because the POS tagging will
 *  be subsequently repeated by another annotating component. However,
 *  this overhead is small because (1) POS tagging is fast compared to dependency
 *  parsing and (2) about half of the tagged documents will be filtered out and,
 *  thus, won't participate in the second stage.
 *  </p>
 * 
 *
 */
public class SimpleTextQualityChecker { 
  /**
   * Annotates the CAS and checks if it's good quality.
   * 
   * @param jcas            An input CAS that will be annotated.
   * @param minTokQty       The minimum number of tokens present to be considered good.
   * @return                true if the CAS contains a high-quality text. 
   * @throws AnalysisEngineProcessException
   */
  public static boolean checkCAS(JCas jcas, int minTokQty) throws AnalysisEngineProcessException {       
    boolean hasNoun = false, hasVerb = false;
    
    for (POS p: JCasUtil.select(jcas, POS.class)) {
      if (p.getPosValue().startsWith("NN")) hasNoun = true;
      if (p.getPosValue().startsWith("VB")) hasVerb = true;
    }
    
    Collection<Token> toks = JCasUtil.select(jcas, Token.class);
    
    return toks.size() >= minTokQty && hasNoun && hasVerb;    
  }  
}

