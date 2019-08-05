/*
 *  Copyright 2015+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.base.Joiner;

public class StringUtils {

  /**
   * Splits the string using the pattern, however, if the input string
   * is empty, it returns the empty array rather than the array with
   * the single empty string.
   * 
   * @param s               input string
   * @param sepPattern      split pattern
   */
  private static String[] emptyArray = new String[0]; 
  
  public static String[] splitNoEmpty(String s, String sepPattern) {
    return s.isEmpty() ? emptyArray : s.split(sepPattern);
  }
  
  /**
   * Checks if a key is in the array by doing a brute-force search:
   * the case is ignored.
   * 
   * @param key   a needle
   * @param arr   a haystack
   * 
   * @return true if the needle is found and false otherwise.
   */
  public static boolean isInArrayNoCase(String key, String [] arr) {
    for (String s: arr) {
      if (s.compareToIgnoreCase(key) == 0) {
        return true;
      }
    }

    return false;
  }
  
  /**
   * Finds a key in the array by doing a brute-force search:
   * the case is ignored.
   * 
   * @param key   a needle
   * @param arr   a haystack
   * 
   * @return a non-negative key index or -1, if the key cannot be found
   */
  public static int findInArrayNoCase(String key, String [] arr) {
    for (int indx = 0; indx < arr.length; ++indx) {
      if (arr[indx].compareToIgnoreCase(key) == 0) {
        return indx;
      }
    }

    return -1;
  }
  
  private static Pattern mReplBR = Pattern.compile("<br\\s*/?>",
      Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);
  private static Pattern mReplTags = Pattern.compile("<[a-z]+[^/>]*/?>",
      Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);

  /**
   * Removes diacritics. Taken from
   * http://www.drillio.com/en/software-development/java/removing-accents-diacritics-in-any-language/.
   */
  public static String removeDiacritics(String text) {
    return text == null ? null : Normalizer.normalize(text, Form.NFD)
        .replaceAll("\\p{InCombiningDiacriticalMarks}+", "");
  }
  
  public static String removePunct(String text) {
    return text == null ? null : text.replaceAll("\\p{Punct}+", " ");
  }

  /**
   * Cleans up a string and replaces diacritcs.
   * 
   * @param s
   * @return
   */
  public static String cleanUp(String s) {
    s = s.trim();
    s = s.replaceAll("\r+", ""); // "\r" may come from a file in DOS encoding;
    s = s.replace('’', '\''); // ugly hack for Yahoo answers

    s = removeDiacritics(s);
    s = s.replaceAll("[^\\x00-\\x7F]", " "); // remove non-ASCII

    /*
     * Repeating punctuation marks cause all kind of trouble in ClearNLP
     * including infinite loops and stack overflow.
     */
    s = s.replaceAll("[?]+", "?");
    s = s.replaceAll("[!]+", "!");
    s = s.replaceAll("[.]+", ".");
    s = s.replaceAll("[:]+", ":");

    Matcher m1 = mReplBR.matcher(s);
    s = m1.replaceAll("\n");
    Matcher m2 = mReplTags.matcher(s);
    return m2.replaceAll(" ").replaceAll("\n+", "\n");
  }

  /**
   * Checks if the string has non-ASCII charaters.
   * 
   * @param id
   * @return
   */
  public static boolean hasNonAscii(String id) {
    for (int i = 0; i < id.length(); ++i) {
      if (id.codePointAt(i) > 127) {
        return true;
      }
    }
    return false;
  }  
  
  /**
   *  Replaces bad Unicode characters, but doesn't change the string length!
   *  <p>
   *  Based on this solution:
   *  http://stackoverflow.com/questions/20762/how-do-you-remove-invalid-hexadecimal-characters-from-an-xml-based-data-source-p
   *  </p>
   *  @param inString   input string
   *  @return an output string (will have the same length as inString).
   */
  public static String removeBadUnicode(String inString) {
    StringBuilder newString = new StringBuilder();
    char ch;

    for (int i = 0; i < inString.length(); i++) {
      ch = inString.charAt(i);
      /*
       *  Replace any characters outside the valid UTF-8 range 
       *  as well as all control characters with the space except 
       *  tabs and new lines.
       *  
       *  Don't delete any chars, we need to preserve word lengths 
       *  and positions!
       *  
       */
      if ((ch < 0x00FD && ch > 0x001F) 
          || ch == '\t' || ch == '\n' || ch == '\r') {
        newString.append(ch);
      } else {
        newString.append(' ');
      }
    }    
    return newString.toString();
  } 

  /**
   * @param  str  a string where replace white-spaces are to be replaced.
   * @return a string with white-spaces replaced by regular spaces.
   */
  public static String replaceWhiteSpaces(String str) {
    return Const.PATTERN_WHITESPACE.matcher(str).replaceAll(" ");
  }

  public static String[] splitOnWhiteSpace(String str) {
    return Const.PATTERN_WHITESPACE.split(str);
  }
  
  public static String joinWithSpace(List<String> arr) {
    return mJoinOnSpace.join(arr);
  }
  
  public static String joinWithSpace(String [] arr) {
    return mJoinOnSpace.join(arr);
  }
  
  static Joiner   mJoinOnSpace  = Joiner.on(' ');
}
