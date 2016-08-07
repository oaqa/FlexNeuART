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
package edu.cmu.lti.oaqa.knn4qa.collection_reader;

import java.io.*;
import java.util.ArrayList;
import java.util.regex.*;
import java.text.Normalizer;
import java.text.Normalizer.Form;

import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.*;
import org.xml.sax.SAXException;
import org.apache.uima.cas.*;
import org.apache.uima.collection.*;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.*;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.utils.*;
import edu.cmu.lti.oaqa.knn4qa.types.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple collection reader that reads Yahoo Answers (from Webscope).
 * It uses the following parameters:
 * 
 * <ul>
 * <li><code>InputFile</code> - path to an (potentially compressed) input file.</li>
 * </ul>
 * 
 * 
 */
public class YahooAnswersReader extends CollectionReader_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(YahooAnswersReader.class);
  
  public static final String DOCUMENT_TAG = "document";
  
  /**
   * Name of the configuration parameter specifying the path of a
   *  potentially compressed input file.
   */
  public static final String PARAM_INPUTFILE = "InputFile";
  /**
   * Name of the configuration parameter specifying the filtering regexp.
   */
  public static final String PARAM_QUEST_FILTER_REGEX = "QuestFilterRegex";
  /**
   * Name of the configuration parameter specifying the maximum number of answers to process.
   */
  public static final String PARAM_MAX_QTY = "MaxQty";
  
  // TODO Can it be in other languages?
  private static final String DOCUMENT_LANGUAGE = "en";
  
  private long          mEstimFileSize = 0;
  private long          mReadByteQty = 0;
  private Pattern       mQuestFilt = null;
  private int           mQty = 0;
  private int           mMaxQty = Integer.MAX_VALUE;
  private XmlIterator   mInpIter;

  private ParsedQuestion mNextDoc = null;

  /**
   * @see org.apache.uima.collection.CollectionReader_ImplBase#initialize()
   */
  public void initialize() throws ResourceInitializationException {
    Integer tmp = (Integer) getConfigParameterValue(PARAM_MAX_QTY);
    if (null != tmp)
      mMaxQty = tmp;
    
    File inpFile = new File(((String) getConfigParameterValue(PARAM_INPUTFILE)).trim());

    // if input directory does not exist or is not a directory, throw exception
    String filePath = inpFile.getPath();

    try {
      InputStream  is = CompressUtils.createInputStream(filePath);
      mEstimFileSize = EstimCompSize.estimUncompSize(filePath);
      mInpIter = new XmlIterator(is, DOCUMENT_TAG);     
    } catch (Exception e) {
      throw new ResourceInitializationException(
          new Exception("Error opening file: '" + filePath + "' exception: " + e.getMessage()));
    }
    
    String fltString = (String) getConfigParameterValue(PARAM_QUEST_FILTER_REGEX);

    if (null != fltString && !fltString.isEmpty()) {
      try {
        mQuestFilt = Pattern.compile(fltString, Pattern.CASE_INSENSITIVE);
      } catch (PatternSyntaxException e) {
        throw new ResourceInitializationException(
            new Exception("Wrong syntax of the filter ('" + PARAM_QUEST_FILTER_REGEX +
                          "' exception: " + e.getMessage()));
      }
    }
    
    try {
      if (mMaxQty > 0) mNextDoc = readNextDoc();
    } catch (Exception e) {
      throw new ResourceInitializationException(e);
    }
  }
  
  private ParsedQuestion readNextDoc() throws Exception {
    while (true) {
      String docText = mInpIter.readNext();
      mReadByteQty += docText.length();
      if (docText.isEmpty()) break;

      ParsedQuestion parsed = null;
      
      try {
        parsed = YahooAnswersParser.parse(docText, true /*do clean up */);
      } catch (Exception e) {      
      // If <bestanswer>...</bestanswer> is missing we may end up here...
      // This is a bit funny, because this element is supposed to be mandatory,
      // but it's not.
        System.err.println("Ignoring... invalid item, exception: " + e);
        continue;
      }
          
      if (mQuestFilt != null) {
        if (!mQuestFilt.matcher(parsed.mQuestion).matches()) continue;
      }
      return parsed;
    }
    return null;
  }

  /**
   * @see org.apache.uima.collection.CollectionReader#hasNext()
   */
  public boolean hasNext() {
    return mNextDoc != null;
  }

  /**
   * @see org.apache.uima.collection.CollectionReader#getNext(org.apache.uima.cas.CAS)
   */
  public void getNext(CAS aCAS) throws IOException, CollectionException {
    if (!hasNext()) {
      throw new CollectionException(new Exception("End of entries!"));
    }
    JCas jcas;
    
    try {
      jcas = aCAS.getJCas();
    } catch (CASException e) {
      throw new CollectionException(e);
    }
    
    StringBuffer sb = new StringBuffer();
    
    int qStart = 0;
    
    sb.append(mNextDoc.mQuestion);
    sb.append(' ');
    sb.append(mNextDoc.mQuestDetail);
    
    int qEnd = sb.length();
    sb.append('\n');
    
    int aStart[]  = new int[mNextDoc.mAnswers.size()];
    int aEnd[]    = new int[mNextDoc.mAnswers.size()];
    
    for (int i = 0; i < mNextDoc.mAnswers.size(); ++i) {
      aStart[i] = sb.length();
      sb.append(mNextDoc.mAnswers.get(i));
      aEnd[i] = sb.length();
      sb.append('\n');
    }
    
    jcas.setDocumentText(sb.toString());
    jcas.setDocumentLanguage(DOCUMENT_LANGUAGE);
    
    Question questAnnot = new Question(jcas, qStart, qEnd);
    questAnnot.setUri(mNextDoc.mQuestUri);
    questAnnot.setBestAnswId(mNextDoc.mBestAnswId);

    ArrayList<Answer> answAnnots = new ArrayList<Answer>();
    for (int i = 0; i < mNextDoc.mAnswers.size(); ++i) {
      Answer  ya = new Answer(jcas, aStart[i], aEnd[i]);
      ya.setIsBest(i == mNextDoc.mBestAnswId);
      ya.addToIndexes();
      ya.setId(mNextDoc.mQuestUri + "-" + i);
      ya.setUri(mNextDoc.mQuestUri);
      answAnnots.add(ya);
    }
    //questAnnot.setAnswers(FSCollectionFactory.createFSArray(jcas, answAnnots)); 
    questAnnot.addToIndexes();
    
    mNextDoc = null;
    ++mQty;
    if (mQty < mMaxQty) {
      try {
        mNextDoc = readNextDoc();
      } catch (Exception e) {
        throw new IOException(e.getMessage());
      }
    }
  }

  /**
   * @see org.apache.uima.collection.base_cpm.BaseCollectionReader#close()
   */
  public void close() throws IOException {
    mInpIter.close();
  }

  /**
   * @see org.apache.uima.collection.base_cpm.BaseCollectionReader#getProgress()
   */
  public Progress[] getProgress() {
    // I don't expect the size in MBs to exceed a billion!
    // TODO it's not clear (1) if this function works properly (2) where is it used?
    long divisor = 1024*1024;
    return new Progress[] {
        new ProgressImpl((int)(Math.min(mEstimFileSize, mReadByteQty)/divisor), 
                          (int)(mEstimFileSize/divisor), Progress.BYTES) 
                          };
  }
}

