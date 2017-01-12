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

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.semantics.type.SemanticArgument;
import de.tudarmstadt.ukp.dkpro.core.api.semantics.type.SemanticPredicate;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;

import edu.cmu.lti.oaqa.knn4qa.types.WNNS;


class GoodTokens {
  HashMap<Token, String>    mMap = new HashMap<Token, String>();
  ArrayList<Token>          mList = new ArrayList<Token>();
}

/**
 * 
 * Produces textual representations of several types: sequence of tokens,
 * bigrams, etc.
 * 
 * @author Leonid Boytsov
 *
 */
public class ExtractTextReps extends ExtractTextRepsBase {
  /**
   * 
   * Constructor.
   * 
   * @param stopWordFileName    the name of the file with stop words.
   * @param bLemmatize          do we lemmatize words?
   * @throws Exception
   */
  ExtractTextReps(String stopWordFileName, boolean bLemmatize) throws Exception {
    super(stopWordFileName, bLemmatize);
  }
  
  /**
   * Produce token lemma, return the original string if the lemma is null;
   * converts the string to lower case.
   * 
   * @param tok
   * @return
   */
  public String getTokenLemma(Token tok) {
    Lemma l = tok.getLemma();
    // For some weird reason, Clear NLP lemma is sometimes NULL
    return (l!=null) ? l.getValue() : tok.getCoveredText().toLowerCase();
  }
  
  /**
   * Retrieves all good (in particular stopwords are excluded) tokens 
   * from a given JCas, for each token obtain respective lemma 
   * (only if mLemmatize is set to true).
   * 
   * @param jCas        input jCas
   * @param isStrict    if true, use a stricter definition of a good term. 
   * @return two things in a single object instance: 
   *         (1) an array of good tokens;
   *         (2) a map, where token object references are mapped to token string values.    
   */
  public GoodTokens getGoodTokens(final JCas jCas, boolean isStrict) {
    GoodTokens res = new GoodTokens();

    for (Token tok : JCasUtil.select(jCas, Token.class)) {
      String text = tok.getCoveredText().toLowerCase();
      
      if (mLemmatize) {
        text = getTokenLemma(tok);
      }
      if (!text.isEmpty() &&
          isGoodWord(text, isStrict) && 
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
  public String getText(final GoodTokens goodToks) {
    StringBuffer sb = new StringBuffer();
    
    for (Token tok : goodToks.mList) {
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
   * Extracts all the good bigrams, bigram words are concatenated using
   * the underscore.
   * 
   * @param goodToks    an object keeping information on all the good tokens.
   * @return a string containing space-separated bigrams.
   */
  public String getBiGram(final GoodTokens goodToks) {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < goodToks.mList.size() - 1; ++i) {
      if (sb.length() > 0) {
        sb.append(' ');
      }
      Token tok1 = goodToks.mList.get(i);
      Token tok2 = goodToks.mList.get(i + 1);
      String str1 = goodToks.mMap.get(tok1);
      String str2 = goodToks.mMap.get(tok2);
      if (null == str1 || null == str2) 
        throw new RuntimeException("Bug: Can't get text of a good token!");
      sb.append(str1 + "_" + str2);
    }

    return strFromStrStream(sb);
  }

  /**
   * Extracts dependency relations (child-parent tokens), the underscore
   * is used to concatenate parts. 
   * 
   * @param jCas            input jCas    
   * @param goodToks        an object keeping information on all the good tokens.
   * @param useLabels       include the argument-type label?
   * @return a string with space-separated dependency relations (with parts 
   *         concatenated using the underscore).
   */
  public String getDepRel(final JCas jCas, final GoodTokens goodToks, 
      boolean useLabels) {
    StringBuffer sb = new StringBuffer();
    
    for (Dependency dep : JCasUtil.select(jCas, Dependency.class)) {
      String strDep = goodToks.mMap.get(dep.getDependent());
      String strGov = goodToks.mMap.get(dep.getGovernor());
      if (strDep == null || strGov == null) // some token isn't good
        continue;
      if (sb.length() > 0) {
        sb.append(' ');
      }
      sb.append(strGov + "_" + (useLabels ? dep.getDependencyType() + "_": "") + strDep);
    }
    
    return strFromStrStream(sb);
  }
  
  /**
   * Extracts all semantic predicates with respective arguments (the underscore
   *          is used to concatenate parts).
   * 
   * @param jCas            input jCas    
   * @param goodToks        an object keeping information on all the good tokens.
   * @param useLabels       include the argument-type label?
   * @return a string with space-separated semantic predicate tuples (with parts 
   *         concatenated using the underscore).
   */
  public String getSrl(final JCas jCas, final GoodTokens goodToks, 
                       boolean useLabels) {
    StringBuffer sb = new StringBuffer();
    
    for (SemanticPredicate pred : JCasUtil.select(jCas, SemanticPredicate.class)) {
      FSArray fa = pred.getArguments();
      for (Token tokPred: JCasUtil.selectCovered(Token.class, pred)) {
        String textPred = goodToks.mMap.get(tokPred);
        if (null == textPred) continue; // e.g. if text1 is a stop word
        for (int i = 0; i < fa.size(); ++i) {
          SemanticArgument arg = (SemanticArgument)fa.get(i); 
          for (Token tokArg: JCasUtil.selectCovered(Token.class, arg)) {
            String textArg = goodToks.mMap.get(tokArg);
            if (null == textArg) continue; // e.g. if text2 is a stop word
            if (sb.length() > 0) {
              sb.append(' ');
            }
            sb.append(textPred + "_" + (useLabels ? arg.getRole() + "_": "") + textArg);
          }
        }
      }
    }
    
    return strFromStrStream(sb);
  }
  
  /**
   * Extract WordNet super senses for non-stop words.
   * 
   * @param jCas    input jCas
   * @return a string of space-separated WNNS belonging to 
   */
  public String getWNSS(final JCas jCas, boolean isStrict) {
    StringBuffer sb = new StringBuffer();
    
    for (WNNS wannot: JCasUtil.select(jCas, WNNS.class)) {
      String text = wannot.getCoveredText().toLowerCase();
      String label = wannot.getSuperSense();
      
      if (!text.isEmpty() &&
          isGoodWord(text, isStrict) && 
          !mStopWords.contains(text)) {
        if (sb.length() > 0) {
          sb.append(' ');
        }
        sb.append(label);
      }
      
    }
    
    return strFromStrStream(sb);
  }  
  
}
