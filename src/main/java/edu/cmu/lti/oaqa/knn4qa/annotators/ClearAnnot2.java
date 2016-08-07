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
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.UimaContext;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import de.tudarmstadt.ukp.dkpro.core.clearnlp.*;

/**
 * Several ClearNLP annotators called in a sequences:
 *  1) Sentence segmentation & tokenization;
 *  2) POS tagging;
 *  3) Dependency parsing;
 *  4) Semantic Role Labeling.
 * 
 * @author Leonid Boytsov
 *
 */
public class ClearAnnot2 extends JCasAnnotator_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(ClearAnnot2.class);
  
  private AnalysisEngine mEngine;

  @Override
  public void initialize(UimaContext context) throws ResourceInitializationException{
    AnalysisEngineDescription aggregate = createEngineDescription(
             // At this point the lemmatizer and segment should have been called
             createEngineDescription(ClearNlpDependencyParser.class,
                                     ClearNlpDependencyParser.PARAM_VARIANT,
                                     "mayo")
            ,createEngineDescription(ClearNlpSemanticRoleLabeler.class,
                                     ClearNlpSemanticRoleLabeler.PARAM_VARIANT,
                                     "mayo")                    
        );

    mEngine = createEngine(aggregate);
    long threadId = Thread.currentThread().getId();
    logger.info(String.format("Created an engine, thread # %d",threadId));
  }

  @Override
  public void process(JCas origJCas) throws AnalysisEngineProcessException {
    PrintInfoHelper.printInfo1(logger, origJCas);
    //logger.info(origJCas.getDocumentText());
    mEngine.process(origJCas);
  }
}

