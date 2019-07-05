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
import java.util.*;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import edu.cmu.lti.oaqa.knn4qa.types.Answer;
import edu.cmu.lti.oaqa.knn4qa.types.Question;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlHelper;
import edu.cmu.lti.oaqa.solr.UtilConst;

/**
 * 
 * This annotator saves answers and questions in the ready-for-indexing format. 
 *
 * @author Leonid Boytsov
 *
 */

public class SolrIndexFileWriter extends JCasAnnotator_ImplBase {  
  private static final String PARAM_INDEX_ANSWER_FILE     = "AnswerFile";
  private static final String PARAM_INDEX_QUESTION_FILE   = "QuestionFile";
  private static final String PARAM_STOPWORD_FILE   = "StopWordFile";
  
  private static final String PARAM_FIELD_TEXT        = "FieldText";   
  private static final String PARAM_FIELD_TEXT_UNLEMM = "FieldTextUnlemm";
  
  private static final XmlHelper mXmlHlp = new XmlHelper();

  private static final boolean STRICTLY_GOOD_TOKENS_FOR_INDEXING = false;
  
  private static int                    mIOState = 0;
  private static BufferedWriter         mIndexQuestionFile;
  private static BufferedWriter         mIndexAnswerFile;
  
  private String mFieldText;
  private String mFieldTextUnlemm;
  
  private ExtractTextReps mTextRepExtract;
  private ExtractTextReps mTextUnlemmRepExtract;

  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);

    String indexQuestionFileName = (String)aContext.getConfigParameterValue(PARAM_INDEX_QUESTION_FILE);
    
    if (null == indexQuestionFileName) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter value: '" + PARAM_INDEX_QUESTION_FILE + "'"));
    }    
    
    
    String indexAnswerFileName = (String)aContext.getConfigParameterValue(PARAM_INDEX_ANSWER_FILE);
    
    if (null == indexAnswerFileName) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter value: '" + PARAM_INDEX_ANSWER_FILE + "'"));
    }    
    
    try {
      initOutput(indexQuestionFileName, indexAnswerFileName);
      
      String tmps = (String)aContext.getConfigParameterValue(PARAM_STOPWORD_FILE);
      if (tmps == null) throw new ResourceInitializationException(
                                    new Exception("Missing parameter '" + PARAM_STOPWORD_FILE + "'"));
      
      mTextRepExtract = new ExtractTextReps(tmps, true);
      mTextUnlemmRepExtract = new ExtractTextReps(tmps, false);
    } catch (Exception e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }    
    
    mFieldText = (String) aContext.getConfigParameterValue(PARAM_FIELD_TEXT);
    if (mFieldText == null) {
      throw new ResourceInitializationException(
          new Exception("Missing parameter '" + PARAM_FIELD_TEXT + "'"));      
    }
    mFieldTextUnlemm=(String)aContext.getConfigParameterValue(PARAM_FIELD_TEXT_UNLEMM);
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
        
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    Map<String, String>  fieldInfo = new HashMap<String, String>();
        
    GoodTokens goodToks = mTextRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_INDEXING);
    GoodTokens goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_INDEXING);
    
    fieldInfo.put(mFieldText, mTextRepExtract.getText(goodToks));
    if (mFieldTextUnlemm != null)
      fieldInfo.put(mFieldTextUnlemm, mTextUnlemmRepExtract.getText(goodToksUnlemm));
    
    Collection<Answer>     colAnsw  = JCasUtil.select(aJCas, Answer.class);
    Collection<Question>   colQuest = JCasUtil.select(aJCas, Question.class);
    
    boolean bErr = false;
    
    if (!colAnsw.isEmpty()) {
      if (colAnsw.size() == 1) {
        Answer ans = colAnsw.iterator().next();
     
        fieldInfo.put(UtilConst.TAG_DOCNO, ans.getId());
    
        try {
          doOutput(mIndexAnswerFile, fieldInfo);
        } catch (Exception e) {
          e.printStackTrace();
          throw new AnalysisEngineProcessException(e);
        }
      } else bErr = true;    
    } else if (!colQuest.isEmpty()) {
      if (colQuest.size() == 1) {
        Question quest = colQuest.iterator().next();
        
        fieldInfo.put(UtilConst.TAG_DOCNO, quest.getUri());
        
        try {
          doOutput(mIndexQuestionFile, fieldInfo);
        } catch (Exception e) {
          e.printStackTrace();
          throw new AnalysisEngineProcessException(e);
        }
      } else bErr = true;
    } else {
      // Yes, we can get an empty CAS, b/c we cannot drop existing one!
      //bErr = true;
    }
    if (bErr) {
      Exception e = new Exception(
          String.format("Bug: bad CAS format, # of questions %d # of answers %d, text: %s",
              colQuest.size(), colAnsw.size(), aJCas.getDocumentText()));
      throw new AnalysisEngineProcessException(e);
    }
  }

  /*
   * All I/O functions are static synchronized, because may be called by multiple threads.
   * To prevent opening/closing twice, we use the mIOState variable.  
   */  
  private static synchronized void initOutput(String indexQuestionFileName, String indexAnswerFileName) throws IOException {
    if (mIOState  != 0) return;
        
    mIndexQuestionFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexQuestionFileName)));
    mIndexAnswerFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexAnswerFileName)));
        
    mIOState = 1;        
  }     
  
  private static synchronized void finishOutput() throws IOException {
    if (mIOState != 1) return;
    mIndexAnswerFile.close();
    mIndexQuestionFile.close();
    mIOState = 2;
  }
  
  private static synchronized void doOutput(BufferedWriter outFile, 
                                            Map<String, String>  fieldInfo) throws Exception {
    outFile.write(mXmlHlp.genXMLIndexEntry(fieldInfo));
    outFile.write(UtilConst.NL);    
  }
   
  
}
