
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

import java.io.*;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import edu.cmu.lti.oaqa.annographix.solr.SolrEvalUtils;
import edu.cmu.lti.oaqa.knn4qa.types.Answer;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;

/**
 * Creates a file with QRELs: the best answer is considered to be the only
 * relevant one. This class is thread-safe.
 * 
 * @author Leonid Boytsov
 */
public class QrelWriter extends JCasAnnotator_ImplBase {
  private static final String PARAM_QREL_FILE = "QrelFile";
  
  private static int                    mIOState = 0;
  private static BufferedWriter         mQrelFile;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    String qrelFileName = (String)aContext.getConfigParameterValue(PARAM_QREL_FILE);
    
    if (null == qrelFileName) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter value: '" + PARAM_QREL_FILE + "'"));
    }
    try {
      initOutput(qrelFileName);
    } catch (IOException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    for (Answer yans: JCasUtil.select(aJCas, Answer.class)) {
      try {
        //doOutput(yans.getUri(), yans.getId(), yans.getIsBest() ? 1:0);
        // Let's consider any answer to be the best answer
        doOutput(yans.getUri(), yans.getId(), 1);
      } catch (IOException e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }
    }

  }
  
  static synchronized private void doOutput(String topicId, String docId, Integer relGrade) 
      throws IOException {
    SolrEvalUtils.saveQrelOneEntry(mQrelFile, topicId, docId, relGrade);    
  }
  
  static synchronized private void initOutput(String qrelFileName) throws IOException {
    if (mIOState  != 0) return;
        
    mQrelFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(qrelFileName)));
    
    mIOState = 1;        
  }
  
  @Override
  public void collectionProcessComplete() throws AnalysisEngineProcessException {
    try {
      finishOutput();
    } catch (IOException e) {
      e.printStackTrace();
      throw new AnalysisEngineProcessException(e);
    }
  }  
  
  static synchronized private void finishOutput() throws IOException {
    if (mIOState != 1) return;
    mQrelFile.close();
    mIOState = 2;
  }  
}
