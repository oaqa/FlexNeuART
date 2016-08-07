/*
 *  Copyright 2014 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.annographix.util;

import java.io.File;
import java.util.HashSet;

import org.apache.commons.io.FileUtils;

/**
 *   A simple class that keeps a dictionary of (stop)words. 
 *   It can initialize the dictionary from a file or string. 
 *   In doing so, it ignores empty lines, or lines that start with '#'.
 *
 *  @author Leonid Boytsov
 */
public class DictNoComments {
  private HashSet<String>   mDict = new HashSet<String>();
  private boolean           mToLower = false;
  
  /**
   * Checks if the dictionary contains a given key.
   * 
   * @param key a string to search for.
   * @return true, if the string is found or false otherwise.
   */
  public boolean contains(String key) {
    if (mToLower) key = key.toLowerCase();
    return mDict.contains(key);
  }
  
  /**
   * Reads dictionary from a file.
   * 
   * @param file        a file object.
   * @param toLower     should we lowercase?
   * @throws Exception
   */
  public DictNoComments(File file, boolean toLower) throws Exception {
    mToLower = toLower;
    for (String s: FileUtils.readLines(file)) {
      processLine(s);
    }    
  }

  /**
   * Initialized dictionary from a multi-line string.
   * 
   * @param text        a text content of the dictionary file (with newlines).
   * @param toLower     should we lowercase?  
   */
  public DictNoComments(String text, boolean toLower) {
    mToLower = toLower;
    for (String s: text.split("[\n\r]+")) {
      processLine(s);
    }      
  }
  
  private void processLine(String s) {
    s = s.trim();
    if (s.isEmpty() ||  s.startsWith("#")) return;
    if (mToLower) s = s.toLowerCase();
    mDict.add(s);    
  }

  /**
   * Add all stopwords from another dictionary.
   * 
   * @param otherDict the dictionary to merge with.
   */
  public void addAll(DictNoComments otherDict) {
    for (String s: otherDict.mDict) {
      if (mToLower) s = s.toLowerCase();
      mDict.add(s);
    }    
  }
}
