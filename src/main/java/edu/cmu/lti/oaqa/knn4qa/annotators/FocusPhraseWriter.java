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

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;

import java.io.*;

import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import edu.cmu.lti.oaqa.knn4qa.collection_reader.SQuADIntermCollectionReader;
import edu.cmu.lti.oaqa.knn4qa.types.FocusPhrase;

/**
 * It is just saves all focus words/phrases previously found by our question analysis module. 
 * 
 * @author Leonid Boytsov
 *
 */
public class FocusPhraseWriter extends JCasAnnotator_ImplBase {
  protected static final String NL = System.getProperty("line.separator");
  
  private static final String PARAM_OUTPUT = "OutputFile";
  
  @ConfigurationParameter(name = PARAM_OUTPUT, mandatory = true)
  private String mOutputFileName;
  
  private BufferedWriter mOutput;

  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    try {
      mOutput = new BufferedWriter(new FileWriter(mOutputFileName));
    } catch (IOException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    JCas questView = null;
    
    try {
      questView = aJCas.getView(SQuADIntermCollectionReader.QUESTION_VIEW);
    } catch (CASException e) {
      throw new AnalysisEngineProcessException(new Exception("No question view in the CAS!"));      
    }
    try {
      for (FocusPhrase fw : JCasUtil.select(questView, FocusPhrase.class)) {
        mOutput.write(fw.getValue());
        mOutput.write(NL);
      }
    } catch (IOException e) {
      e.printStackTrace();
      throw new AnalysisEngineProcessException(e);
    }
  }
  
  @Override
  public void collectionProcessComplete() throws AnalysisEngineProcessException {
    try {
      mOutput.close();
    } catch (IOException e) {
      e.printStackTrace();
      throw new AnalysisEngineProcessException(e);
    }
  }   

}
