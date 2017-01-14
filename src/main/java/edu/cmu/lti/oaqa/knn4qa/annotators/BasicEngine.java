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

import java.util.ArrayList;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
//import de.tudarmstadt.ukp.dkpro.core.clearnlp.*;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.*;


/**
 * Basic annotation + empty CAS provider.
 * 
 * @author Leonid Boytsov
 *
 */
public class BasicEngine {
  private static final Logger logger = LoggerFactory.getLogger(BasicEngine.class);
  private static int mThreadQty = 0;

  public BasicEngine(UimaContext context, boolean doPOSTagging) throws ResourceInitializationException{
    AnalysisEngineDescription aggregate = null;
    if (doPOSTagging) {
      aggregate = createEngineDescription(
  // The specified Stanford model and the ClearNLP model result
  // in almost identical tagging speed of 6-8K tokens per second with 3-4 threads
  // Yet, Stanford sentence segmentation, lemmatization, and tokenization seem to provide better results.
          
  //        createEngineDescription(ClearNlpSegmenter.class),
  //        createEngineDescription(ClearNlpPosTagger.class, ClearNlpPosTagger.PARAM_VARIANT, "mayo"),
  //        createEngineDescription(ClearNlpLemmatizer.class)
        createEngineDescription(StanfordSegmenter.class),
        createEngineDescription(StanfordLemmatizer.class),
        createEngineDescription(StanfordPosTagger.class, StanfordPosTagger.PARAM_VARIANT, "left3words-distsim")  
          );
      logger.info("Using POS tagger");
    } else {
      aggregate = createEngineDescription(
        createEngineDescription(StanfordSegmenter.class),
        createEngineDescription(StanfordLemmatizer.class)
        );
      logger.info("Not using POS tagger");
    }

    mEngine = createEngine(aggregate);
    int tqty = 0;
    synchronized (this.getClass()) {
      mThreadQty ++;
      tqty = mThreadQty;
    }
    logger.info(String.format("Created an engine, # of threads %d", tqty));
  }

  public void process(JCas origJCas) throws AnalysisEngineProcessException {
    mEngine.process(origJCas);
  }
  
  public JCasFactory createJCasFactory() {
    return new JCasFactory(mEngine);
  }
  
  private AnalysisEngine   mEngine;
}


