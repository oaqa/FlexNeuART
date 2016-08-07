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

import java.io.IOException;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_component.JCasAnnotator_ImplBase;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import edu.cmu.lti.oaqa.knn4qa.utils.WordNetUtil;
import edu.cmu.lti.oaqa.knn4qa.types.WNNS;

/**
 * Creates WordNet super sense annotations (for nouns, adjectives, verbs, and adverbs).
 * 
 * @author Leonid Boytsov
 *
 */
public class WordNetSuperSenseAnnot extends JCasAnnotator_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(WordNetSuperSenseAnnot.class);
  
  public final String   WORDNET_HOME    = "WordNetHome";

  WordNetUtil           mWordNet;
  
  @Override
  public void initialize(UimaContext aContext)
  throws ResourceInitializationException {
    super.initialize(aContext);
    
    String wnh = (String) aContext.getConfigParameterValue(WORDNET_HOME);
    
    if (wnh == null) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter : '" 
                         + WORDNET_HOME + "'"));                  
    }

    try {
      mWordNet = new WordNetUtil(wnh, true /* let's use lexical name for clarity */);
    } catch (IOException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }    
  }

  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    try {
      for (POS pos : JCasUtil.select(aJCas, POS.class)) {
        String text = pos.getCoveredText().toLowerCase();
        String posTag = pos.getPosValue();
        String wnss = "";
        try {
          wnss = mWordNet.getLexName(text, posTag);
        } catch (IllegalArgumentException e) {
          logger.warn("Failed to process word '" + text + "' tag: '" + posTag);
        }

        if (!wnss.isEmpty()) {
          WNNS annot = new WNNS(aJCas, pos.getBegin(), pos.getEnd());
          annot.setSuperSense(wnss);
          annot.addToIndexes();
          //System.out.println(annot.getSuperSense() + " " + annot.getBegin() + ":" + annot.getEnd());
        }
      }
    } catch (Exception e) {
      throw new AnalysisEngineProcessException(e);
    }
  }
}
