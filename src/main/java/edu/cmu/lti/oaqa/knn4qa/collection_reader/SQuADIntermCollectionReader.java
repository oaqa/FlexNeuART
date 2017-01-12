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
package edu.cmu.lti.oaqa.knn4qa.collection_reader;

import java.io.*;

import org.apache.uima.UimaContext;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.component.CasCollectionReader_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Progress;
import org.apache.uima.util.ProgressImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;

import edu.cmu.lti.oaqa.knn4qa.qaintermform.QAData;
import edu.cmu.lti.oaqa.knn4qa.qaintermform.QAPassage;
import edu.cmu.lti.oaqa.knn4qa.types.*;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;

/**
 * 
 * A collection reader that needs SQuAD-like data converted to an intermediate format:
 * note that it REMOVES DIACRITICAL signs.
 * 
 * @author Leonid Boytsov
 *
 */
public class SQuADIntermCollectionReader extends CasCollectionReader_ImplBase {
  public  static final String QUESTION_VIEW = "QuestionView";
  // TODO Can it be in other languages?
  public static final String DOCUMENT_LANGUAGE = "en";
  
  private static final Logger logger = LoggerFactory.getLogger(SQuADIntermCollectionReader.class);
  
  
  Gson mGSON = new Gson();
  
  private int             mProcCASQty;
  private int             mProcDocQty;
  private int             mPassageQty;
  private int             mPassageIndex;
  private boolean         mEOF;
  private BufferedReader  mInput;
  private QAData          mTmpData;
  
  /**
   * Name of the configuration parameter specifying the path of a
   *  potentially compressed input file.
   */
  private static final String PARAM_INPUTFILE = "InputFile";
  private static final String PARAM_MAXQTY = "MaxQty";
 
  
  @ConfigurationParameter(name = PARAM_INPUTFILE, mandatory = true)
  private String mInputFileName;  
  
  @ConfigurationParameter(name = PARAM_MAXQTY, mandatory = false)
  private Integer mMaxQty = Integer.MAX_VALUE;

  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    logger.info("Input file: " + mInputFileName);
    logger.info("Maxim # of input documents to process: " + mMaxQty);

    mProcCASQty = 0;
    mProcDocQty = 0;
    
    mPassageQty = 0;
    mPassageIndex = 0;
    mEOF = false;
    
    try {
      mInput = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(mInputFileName)));
    } catch (IOException io) {
      throw new ResourceInitializationException(io);
    }
  }

  @Override
  public void getNext(CAS aCAS) throws IOException, CollectionException {
    //System.out.println(mProcDocQty + " -> " + mProcCASQty);
    if (!hasNext()) 
      throw new IOException("Reading beyond EOF");
    // Here the following should hold:
    // mEOF == false && mPassageIndex < mPassageQty
    JCas jcas;
    
    try {
      jcas = aCAS.getJCas();
    } catch (CASException e) {
      throw new CollectionException(e);
    }    
    
    QAPassage currPass = mTmpData.passages[mPassageIndex++];
    
    jcas.setDocumentText(StringUtilsLeo.removeDiacritics(currPass.text));
    jcas.setDocumentLanguage(DOCUMENT_LANGUAGE);
    
    JCas jcasQuest;
    
    try {    
      jcasQuest = jcas.createView(QUESTION_VIEW);
    } catch (CASException e) {
      throw new CollectionException(e);
    }
    
    StringBuffer sb = new StringBuffer();       
    
    if (currPass.questions != null) {
      int          qlen     = currPass.questions.length;
      int          qStart[] = new int[qlen];
      int          qEnd[]   = new int[qlen]; 
      
      int          len = 0;
      // Create multiple question annotations, including question IDs
      for (int qid = 0; qid < qlen; ++ qid) {
        qStart[qid] = len;
        String qtext = StringUtilsLeo.removeDiacritics(currPass.questions[qid].text);
        len += 1 + qtext.length();
        qEnd[qid] = len;
        sb.append(qtext); sb.append(' ');
      }
      
      for (int qid = 0; qid < qlen; ++ qid) {
        FactoidQuestion q  = new FactoidQuestion(jcasQuest, qStart[qid], qEnd[qid]-1);
        q.setId(currPass.questions[qid].id);
        q.addToIndexes();
      }
    }
    
    // Fill out the question JCAS even if no questions are given
    jcasQuest.setDocumentLanguage(DOCUMENT_LANGUAGE);
    jcasQuest.setDocumentText(sb.toString());
    
    Passage p = new Passage(jcas, 0, jcas.getDocumentText().length());
    p.setId(currPass.id);
    p.addToIndexes();
    
    mProcCASQty++;
  }

  @Override
  public boolean hasNext() throws IOException, CollectionException {
    if (mEOF) return false;
    if (mPassageIndex < mPassageQty) return true;
    readNextLine();
    return !mEOF;
  }

  private void readNextLine() throws IOException {
    do {
      String line = mInput.readLine();

      if (line == null) {
        mEOF = true;
        mPassageQty = mPassageIndex = 0;
        return;
      }
      //System.out.println(line);
      mTmpData = mGSON.fromJson(line, QAData.class);
      
      mPassageIndex = 0;
      mPassageQty = 0;
      if (mTmpData.passages != null) 
        mPassageQty = mTmpData.passages.length;
    } while (mPassageQty == 0); // this would skip all empty records with no passages
    // Here mPassageQty > 0
    if (++mProcDocQty > mMaxQty) {
      mEOF = true;
    }
    
  }


  @Override
  public Progress[] getProgress() {
    return new Progress[]{new ProgressImpl(mProcCASQty, -1, Progress.ENTITIES)};
  }

  @Override
  public void close() throws IOException {
    mInput.close();
  }

}
