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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.uima.cas.text.AnnotationFS;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import edu.cmu.lti.oaqa.knn4qa.types.*;
import edu.cmu.lti.oaqa.knn4qa.utils.DictNoComments;

class GoodTokensSQuAD {
  HashMap<TokenLemma, String>    mMap = new HashMap<TokenLemma, String>();
  ArrayList<TokenLemma>          mList = new ArrayList<TokenLemma>();
}

public class ExtractTextRepsSQuAD extends ExtractTextRepsBase {

  private DictNoComments mFreqFocusWords = null;

  ExtractTextRepsSQuAD(String stopWordFileName, String freqFocusWordFileName, boolean bLemmatize) throws Exception {
    super(stopWordFileName, bLemmatize);
    mFreqFocusWords = new DictNoComments(new File(freqFocusWordFileName), true /* lowercasing */);
  }
  
  /**
   * Retrieves all good (in particular stopwords are excluded) tokens 
   * from a given JCas, which are "covered" by a given annotation;
   * for each token obtain respective lemma 
   * (only if mLemmatize is set to true).
   * 
   * @param jCas        input jCas
   * @param coverAnnot  covering annotation 
   * @param isStrict    if true, use a stricter definition of a good term. 
   * @return two things in a single object instance: 
   *         (1) an array of good tokens;
   *         (2) a map, where token object references are mapped to token string values.    
   */
  public GoodTokensSQuAD getGoodTokens(final JCas jCas, AnnotationFS coverAnnot) {
    GoodTokensSQuAD res = new GoodTokensSQuAD();

    for (TokenLemma tok : JCasUtil.selectCovered(jCas, TokenLemma.class, coverAnnot)) {     
      String text = tok.getCoveredText().toLowerCase();
      
      if (mLemmatize) {
        text = tok.getLemma();
      }
      if (!text.isEmpty() &&
          // isGoodWord is too restrictive
          //isGoodWord(text, isStrict) && 
          !startsWithPunct(text) &&
          !mStopWords.contains(text)) 
      {
        res.mMap.put(tok, text);
        res.mList.add(tok);
      }
    }       
    
    return res;
  }  
  
  /**
   * Generate text from the list of good tokens.
   * 
   * @param goodToks    an object keeping information on all the good tokens.
   * @return a string containing space-separated tokens.
   */
  public String getText(final GoodTokensSQuAD goodToks) {
    StringBuffer sb = new StringBuffer();
    
    for (TokenLemma tok : goodToks.mList) {
      if (sb.length() > 0) {
        sb.append(' ');
      }
      String str = goodToks.mMap.get(tok);
      if (null == str) 
        throw new RuntimeException("Bug: Can't get text of a good token!");      
      sb.append(str);
    }

    return strFromStrStream(sb);    
  }
  
  /**
   * Generate a list of named entities from the passage annotation
   * 
   * @param jCas  a JCas containing entity annotations
   * @return
   */
  public String getNER(final JCas jCas) {
    StringBuffer sb = new StringBuffer();
    
    for (Entity e : JCasUtil.select(jCas, Entity.class)) {
      sb.append(' ');
      sb.append(e.getEtype() + ":" + e.getLabel());
    }        
    
    return strFromStrStream(sb);
  }
  
  /**
   * Generate a list of question annotations from the respective annotated factoid question.
   * 
   * @param questView     Question view jCas
   * @param coverAnnot    Covering annotation
   * @param bWWord        Should we include question words 
   * @param bFocusWord    Should we include common focus words?
   * @param bEpyraQType   Should we include common feature types
   * @return
   */
  public String getQuestionAnnot(final JCas questView, FactoidQuestion coverAnnot, 
                                  boolean bWWord,
                                  boolean bFocusWord,
                                  boolean bEpyraQType
                                 ) {
    StringBuffer sb = new StringBuffer();
    
    if (bWWord) {
      for (WWord ww: JCasUtil.selectCovered(questView, WWord.class, coverAnnot)) {
         sb.append(' ');
         sb.append(ww.getValue());
      }
    }
    
    if (bFocusWord) {
      for (FocusPhrase fw: JCasUtil.selectCovered(questView, FocusPhrase.class, coverAnnot)) {
         sb.append(' ');
         String focusWord = fw.getValue().toLowerCase();
         if (mFreqFocusWords == null || mFreqFocusWords.contains(focusWord))
           sb.append(focusWord);
      }
    }
    
    if (bEpyraQType) {
      for (FactoidQuestionType qt: JCasUtil.selectCovered(questView, FactoidQuestionType.class, coverAnnot)) {
        sb.append(' ');
        sb.append(qt.getValue());
      }      
    }
    
    return strFromStrStream(sb);
  }
  
}
