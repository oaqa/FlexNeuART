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
import java.util.regex.Pattern;

import com.google.common.base.CharMatcher;

import edu.cmu.lti.oaqa.knn4qa.utils.DictNoComments;

public class ExtractTextRepsBase {

  protected boolean mLemmatize = false;
  protected DictNoComments mStopWords;

  /**
   * 
   * Constructor.
   * 
   * @param stopWordFileName    the name of the file with stop words.
   * @param bLemmatize          do we lemmatize words?
   * @throws Exception
   */
  ExtractTextRepsBase(String stopWordFileName, boolean bLemmatize) throws Exception {
    mStopWords = new DictNoComments(new File(stopWordFileName), true /* lowercasing */);
    mLemmatize = bLemmatize;
  }

  /**
   * Obtain a string from stream, remove trailing '\n', '\r'.
   * 
   * @param sb  input string stream.
   * @return resulting string.
   */
  protected String strFromStrStream(StringBuffer sb) {
    String tmp = sb.toString();
    return tmp.replace('\n', ' ').replace('\r', ' ').trim();
  }

  /**
   * 
   * A good word should start from the letter: it may contain a letter,
   * a dash, or an apostrophe. 
   * 
   * @param text        input
   * @param isStrict    if true, use a stricter definition of a good term.    
   * @return true if a good word.
   */
  protected boolean isGoodWord(String text, boolean isStrict) {
    if (text.isEmpty()) return false;
    CharMatcher m = isStrict ? CharMatcher.JAVA_LETTER : 
                               CharMatcher.JAVA_LETTER_OR_DIGIT;
    if (!m.matches(text.charAt(0))) return false;
    for (int i = 0; i < text.length(); ++i) {
      char c = text.charAt(i);
      if (c != '-' && c != '\'' &&  !m.matches(c)) {
        return false;
      }
    }
        
    return true;
  }
  
  protected static boolean startsWithPunct(String text) {
    return Pattern.matches("^\\p{Punct}.*$", text);
  }
}