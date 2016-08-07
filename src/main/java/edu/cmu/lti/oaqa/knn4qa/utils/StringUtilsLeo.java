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
package edu.cmu.lti.oaqa.knn4qa.utils;

public class StringUtilsLeo {
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
}
