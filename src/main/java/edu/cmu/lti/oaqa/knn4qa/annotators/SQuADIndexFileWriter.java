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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;
import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.SQuADIntermCollectionReader;
import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.types.FactoidQuestion;
import edu.cmu.lti.oaqa.knn4qa.types.Passage;

/**
 * This annotator saves passages and questions in the ready-for-indexing format.
 * 
 * @author Leonid Boytsov
 *
 */
public class SQuADIndexFileWriter extends JCasAnnotator_ImplBase {
  protected static final String NL = System.getProperty("line.separator");
  
  private static final String PARAM_INDEX_PASSAGE_FILE    = "PassageFile";
  private static final String PARAM_INDEX_QUESTION_FILE   = "QuestionFile";
  private static final String PARAM_STOPWORD_FILE         = "StopWordFile";
  
  private static final String PARAM_FIELD_TEXT            = "FieldText"; 
  private static final String PARAM_FIELD_TEXT_UNLEMM     = "FieldTextUnlemm";
    
  private static final XmlHelper mXmlHlp = new XmlHelper();
  
  private static BufferedWriter         mIndexQuestionFile;
  private static BufferedWriter         mIndexPassageFile;
  private static int                    mIOState = 0;
 
  @ConfigurationParameter(name = PARAM_INDEX_QUESTION_FILE, mandatory = true)
  private String mIndexQuestionFileName;
  @ConfigurationParameter(name = PARAM_INDEX_PASSAGE_FILE, mandatory = true)
  private String mIndexPassageFileName;
  @ConfigurationParameter(name = PARAM_STOPWORD_FILE, mandatory = true)
  private String mStopWordFileName;
  
  private String mFieldText;
  private String mFieldTextUnlemm;
  
  private SQuADExtractTextReps mTextRepExtract;
  private SQuADExtractTextReps mTextUnlemmRepExtract;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    try {
      initOutput(mIndexQuestionFileName, mIndexPassageFileName);
      
      mTextRepExtract = new SQuADExtractTextReps(mStopWordFileName, true);
      mTextUnlemmRepExtract = new SQuADExtractTextReps(mStopWordFileName, false);
      
      mFieldText = (String) aContext.getConfigParameterValue(PARAM_FIELD_TEXT);
      if (mFieldText == null) {
        throw new ResourceInitializationException(
            new Exception("Missing parameter '" + PARAM_FIELD_TEXT + "'"));      
      }
      mFieldTextUnlemm=(String)aContext.getConfigParameterValue(PARAM_FIELD_TEXT_UNLEMM);
      if (mFieldTextUnlemm == null) {
        throw new ResourceInitializationException(
            new Exception("Missing parameter '" + PARAM_FIELD_TEXT_UNLEMM + "'"));      
      }
    } catch (Exception e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }    
  }  
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {    
  
    
    // 1. Process the passage
    {
      Map<String, String>  fieldInfo = new HashMap<String, String>();
      
      Passage passage = JCasUtil.selectSingle(aJCas, Passage.class); // Will throw an exception of Passage is missing
      
      {
        GoodTokensSQuAD goodToks = mTextRepExtract.getGoodTokens(aJCas, passage);   
        String text = mTextRepExtract.getText(goodToks);
        fieldInfo.put(mFieldText, text);
      }
      if (mFieldTextUnlemm != null) {
        GoodTokensSQuAD goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, passage);
        String textUnlemm = mTextUnlemmRepExtract.getText(goodToksUnlemm);
        fieldInfo.put(mFieldTextUnlemm, textUnlemm);
      }

      fieldInfo.put(UtilConst.TAG_DOCNO, passage.getId());
       
      try {
        doOutput(mIndexPassageFile, fieldInfo);
      } catch (Exception e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }

    }
    // 2. Process the questions if any are present
    {
      JCas questView = null;
      try {
        questView = aJCas.getView(SQuADIntermCollectionReader.QUESTION_VIEW);
      } catch (CASException e) {
        throw new AnalysisEngineProcessException(new Exception("No question view in the CAS!"));      
      }
      
      for (FactoidQuestion q : JCasUtil.select(questView, FactoidQuestion.class)) {
        Map<String, String>  fieldInfo = new HashMap<String, String>();        

        fieldInfo.put(UtilConst.TAG_DOCNO, q.getId());

        {
          GoodTokensSQuAD goodToks = mTextRepExtract.getGoodTokens(aJCas, q);   
          String text = mTextRepExtract.getText(goodToks);
          fieldInfo.put(mFieldText, text);
        }
        if (mFieldTextUnlemm != null) {
          GoodTokensSQuAD goodToksUnlemm = mTextUnlemmRepExtract.getGoodTokens(aJCas, q);
          String textUnlemm = mTextUnlemmRepExtract.getText(goodToksUnlemm);
          fieldInfo.put(mFieldTextUnlemm, textUnlemm);
        }
  
        
        try {
          doOutput(mIndexQuestionFile, fieldInfo);
        } catch (Exception e) {
          e.printStackTrace();
          throw new AnalysisEngineProcessException(e);
        }
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
  private static synchronized void initOutput(String indexQuestionFileName, String indexPassageFileName) 
                                                                                                    throws IOException {
    if (mIOState  != 0) return;
        
    mIndexQuestionFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexQuestionFileName)));
    mIndexPassageFile = new BufferedWriter(
        new OutputStreamWriter(CompressUtils.createOutputStream(indexPassageFileName)));
        
    mIOState = 1;        
  }
  private static synchronized void finishOutput() throws IOException {
    if (mIOState != 1) return;
    mIndexPassageFile.close();
    mIndexQuestionFile.close();
    mIOState = 2;
  }
  
  private static synchronized void doOutput(BufferedWriter outFile, Map<String, String>  fieldInfo) throws Exception {
    outFile.write(mXmlHlp.genXMLIndexEntry(fieldInfo));
    outFile.write(NL);
  }

}
