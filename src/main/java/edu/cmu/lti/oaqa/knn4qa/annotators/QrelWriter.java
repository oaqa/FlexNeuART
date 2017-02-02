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
  private static final String PARAM_QREL_FILE_PREFIX = "QrelFilePrefix";
  
  public static final int QREL_BEST_GRADE  = 4;
  public static final int QREL_OTHER_GRADE = 3;
  
  private static int                    mIOState = 0;
  private static BufferedWriter         mQrelFileBinary;
  private static BufferedWriter         mQrelFileGraded;
  private static BufferedWriter         mQrelFileOnlyBest;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    String qrelFileNamePrefix = (String)aContext.getConfigParameterValue(PARAM_QREL_FILE_PREFIX);
    
    if (null == qrelFileNamePrefix) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter value: '" + PARAM_QREL_FILE_PREFIX + "'"));
    }
    try {
      initOutput(qrelFileNamePrefix);
    } catch (IOException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
  }
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    for (Answer yans: JCasUtil.select(aJCas, Answer.class)) {
      try {
        doOutput(mQrelFileBinary,   yans.getUri(), yans.getId(), 1);
        doOutput(mQrelFileGraded,   yans.getUri(), yans.getId(), yans.getIsBest() ? QREL_BEST_GRADE:QREL_OTHER_GRADE);
        doOutput(mQrelFileOnlyBest, yans.getUri(), yans.getId(), yans.getIsBest() ? QREL_BEST_GRADE:0);
      } catch (IOException e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }
    }

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
  
  /*
   * All I/O functions are static synchronized, because may be called by multiple threads.
   * To prevent opening/closing twice, we use the mIOState variable.  
   */  

  static synchronized private void initOutput(String qrelFileNamePrefix) throws IOException {
    if (mIOState  != 0) return;
        
    mQrelFileBinary = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(qrelFileNamePrefix + "_all_binary.txt")));
    mQrelFileGraded = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(qrelFileNamePrefix + "_all_graded.txt")));
    mQrelFileOnlyBest = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(qrelFileNamePrefix + "_onlybest.txt")));    
    
    
    mIOState = 1;        
  }
    
  static synchronized private void finishOutput() throws IOException {
    if (mIOState != 1) return;
    mQrelFileBinary.close();
    mQrelFileGraded.close();
    mQrelFileOnlyBest.close();
    mIOState = 2;
  }    
  
  static synchronized private void doOutput(BufferedWriter qrelFile, String topicId, String docId, Integer relGrade) 
      throws IOException {
    SolrEvalUtils.saveQrelOneEntry(qrelFile, topicId, docId, relGrade);    
  }  
}
