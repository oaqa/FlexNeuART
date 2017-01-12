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

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.types.Answer;
import edu.cmu.lti.oaqa.knn4qa.types.Question;

/**
 * 
 * This annotator saves answers and questions in the ready-for-indexing format. 
 *
 * @author Leonid Boytsov
 *
 */

public class SolrIndexFileWriter extends JCasAnnotator_ImplBase {
  protected static final String NL = System.getProperty("line.separator");
  
  private static final String PARAM_INDEX_ANSWER_FILE     = "AnswerFile";
  private static final String PARAM_INDEX_QUESTION_FILE   = "QuestionFile";
  private static final String PARAM_STOPWORD_FILE   = "StopWordFile";
  
  private static final String PARAM_FIELD_TEXT        = "FieldText";   
  private static final String PARAM_FIELD_TEXT_UNLEMM = "FieldTextUnlemm";
  private static final String PARAM_FIELD_BIGRAM      = "FieldBiGram"; 
  private static final String PARAM_FIELD_DEP_REL     = "FieldDepRel"; 
  private static final String PARAM_FIELD_SRL         = "FieldSrl";    
  private static final String PARAM_FIELD_SRL_LAB     = "FieldSrlLab"; 
  private static final String PARAM_FIELD_WNNS        = "FieldWNSS"; // Can be omitted
  
  
  private static final XmlHelper mXmlHlp = new XmlHelper();

  private static final boolean STRICTLY_GOOD_TOKENS_FOR_INDEXING = false;
  
  private static int                    mIOState = 0;
  private static BufferedWriter         mIndexQuestionFile;
  private static BufferedWriter         mIndexAnswerFile;
  
  private String mFieldText;
  private String mFieldTextUnlemm;
  private String mFieldBiGram;
  private String mFieldDepRel;
  private String mFieldSrl;
  private String mFieldSrlLab;
  private String mFieldWNNS;
  
  private ExtractTextReps mTextRepExtract;
  private ExtractTextReps mTextUnlemmRepExtract;

  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    Boolean tmpb = null;
    
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
    mFieldBiGram = (String) aContext.getConfigParameterValue(PARAM_FIELD_BIGRAM);
    mFieldDepRel = (String) aContext.getConfigParameterValue(PARAM_FIELD_DEP_REL);
    mFieldSrl    = (String) aContext.getConfigParameterValue(PARAM_FIELD_SRL);
    mFieldSrlLab = (String) aContext.getConfigParameterValue(PARAM_FIELD_SRL_LAB);
    mFieldWNNS   = (String) aContext.getConfigParameterValue(PARAM_FIELD_WNNS);
  }

  static synchronized private void initOutput(String indexQuestionFileName, String indexAnswerFileName) throws IOException {
    if (mIOState  != 0) return;
        
    mIndexQuestionFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexQuestionFileName)));
    mIndexAnswerFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexAnswerFileName)));
        
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
    mIndexAnswerFile.close();
    mIndexQuestionFile.close();
    mIOState = 2;
  }      
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    Map<String, String>  fieldInfo = new HashMap<String, String>();
        
    GoodTokens goodToks = mTextRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_INDEXING);
    GoodTokens goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, STRICTLY_GOOD_TOKENS_FOR_INDEXING);
    
    fieldInfo.put(mFieldText, mTextRepExtract.getText(goodToks));
    if (mFieldTextUnlemm != null)
      fieldInfo.put(mFieldTextUnlemm, mTextUnlemmRepExtract.getText(goodToksUnlemm));
    if (mFieldBiGram != null)
      fieldInfo.put(mFieldBiGram, mTextRepExtract.getBiGram(goodToks));
    if (mFieldDepRel != null)
      fieldInfo.put(mFieldDepRel, mTextRepExtract.getDepRel(aJCas, goodToks, false /* without labels */));
    if (mFieldSrl != null)
      fieldInfo.put(mFieldSrl, mTextRepExtract.getSrl(aJCas, goodToks, false /* without labels */));
    if (mFieldSrlLab != null)
      fieldInfo.put(mFieldSrlLab, mTextRepExtract.getSrl(aJCas, goodToks, true /* with labels */)); 
    if (mFieldWNNS != null)
      fieldInfo.put(mFieldWNNS, mTextRepExtract.getWNSS(aJCas, STRICTLY_GOOD_TOKENS_FOR_INDEXING));    
    
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

  static synchronized private void doOutput(BufferedWriter outFile, 
                                            Map<String, String>  fieldInfo) throws Exception {
    outFile.write(mXmlHlp.genXMLIndexEntry(fieldInfo));
    outFile.write(NL);    
  }
  
}
