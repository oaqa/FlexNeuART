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
  private static final String PARAM_FOCUSWORD_FILE        = "FreqFocusWordFile";
    
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
  // Let this one be mandatory, there's no harm reading a small file in every SQuAD pipeline
  @ConfigurationParameter(name = PARAM_FOCUSWORD_FILE, mandatory = true) 
  private String mFocusWordFile;
  
  private SQuADExtractTextReps mTextRepExtract;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    try {
      initOutput(mIndexQuestionFileName, mIndexPassageFileName);
      
      mTextRepExtract = new SQuADExtractTextReps(mStopWordFileName, mFocusWordFile, true);
    } catch (Exception e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }    
  }  
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {    
    String textSOLR = FeatureExtractor.mFieldsSOLR[FeatureExtractor.TEXT_FIELD_ID];
    
    // 1. Process the passage
    {
      Map<String, String>  fieldInfo = new HashMap<String, String>();
      
      Passage passage = JCasUtil.selectSingle(aJCas, Passage.class); // Will throw an exception of Passage is missing
      
      GoodTokensSQuAD goodToks = mTextRepExtract.getGoodTokens(aJCas, passage);      
      
      String text       = mTextRepExtract.getText(goodToks);
      String allNER     = mTextRepExtract.getNER(aJCas, passage, true, true);
      String dbpediaNER = mTextRepExtract.getNER(aJCas, passage, true, false);
      String spacyNER   = mTextRepExtract.getNER(aJCas, passage, false, true);

      fieldInfo.put(UtilConst.TAG_DOCNO, passage.getId());
       
      fieldInfo.put(textSOLR, text);
      fieldInfo.put(FeatureExtractor.TEXT_QFEAT, text + " " + allNER);
      fieldInfo.put(FeatureExtractor.QFEAT_ONLY, allNER);
      
      fieldInfo.put(FeatureExtractor.EPHYRA_ALLENT,    allNER);
      fieldInfo.put(FeatureExtractor.EPHYRA_DBPEDIA,   dbpediaNER);
      fieldInfo.put(FeatureExtractor.EPHYRA_SPACY,     spacyNER);
      fieldInfo.put(FeatureExtractor.LEXICAL_ALLENT,   allNER);
      fieldInfo.put(FeatureExtractor.LEXICAL_DBPEDIA,  dbpediaNER);
      fieldInfo.put(FeatureExtractor.LEXICAL_SPACY,    spacyNER);
      
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
        
        GoodTokensSQuAD goodToks = mTextRepExtract.getGoodTokens(questView, q);
        String text = mTextRepExtract.getText(goodToks);
        // Lexical question features: w-word and focus word/phrase
        String qLexical = mTextRepExtract.getQuestionAnnot(questView, q, 
                                              true /* w-word */, true /* focus word */, false /* Ephyra type */);
        // Ephyra question type
        String qType = mTextRepExtract.getQuestionAnnot(questView, q, 
                                              false /* w-word */, false /* focus word */, true /* Ephyra type */);
        
        fieldInfo.put(UtilConst.TAG_DOCNO, q.getId());

        fieldInfo.put(textSOLR, text);
        fieldInfo.put(FeatureExtractor.TEXT_QFEAT, qLexical + " " + text + " " + qType);
        fieldInfo.put(FeatureExtractor.QFEAT_ONLY, qLexical + " " + qType);
        
        fieldInfo.put(FeatureExtractor.EPHYRA_ALLENT,    qType);
        fieldInfo.put(FeatureExtractor.EPHYRA_DBPEDIA,   qType);
        fieldInfo.put(FeatureExtractor.EPHYRA_SPACY,     qType);
        fieldInfo.put(FeatureExtractor.LEXICAL_ALLENT,   qLexical);
        fieldInfo.put(FeatureExtractor.LEXICAL_DBPEDIA,  qLexical);
        fieldInfo.put(FeatureExtractor.LEXICAL_SPACY,    qLexical);
        
        
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
