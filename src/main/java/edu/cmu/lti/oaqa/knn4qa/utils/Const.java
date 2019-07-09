/*
 *  Copyright 2014+ Carnegie Mellon University
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

import java.util.regex.Pattern;

/**
 * 
 * The file with various Solr-related constants and helper functions.
 * 
 * @author Leonid Boytsov
 *
 */
public class Const {
  public static final String ENCODING_NAME = "UTF-8";
  public static final String USER_AGENT = "Mozilla/4.0";
  
  /**
   * An XML version, must be 1.0.
   */
  public static final String XML_VERSION = "1.0";
  
  public static final String TAG_DOC_ENTRY = "DOC";
  public static final String TAG_DOCNO     = "DOCNO";
  public static final String TEXT_FIELD_NAME = "text";
  
  public static final String NL = System.getProperty("line.separator");

  /** These are all ASCII punctuation chars except the apostrophe! */
  public static final String NON_INDEXABLE_PUNCT = 
                                    "!\"#$%&()*+,-./:;<=>?@\\[\\]^_`{¦}~\\\\";
  
  /* A pesky stop-word arising as a result of tokenization */
  public static final String PESKY_STOP_WORD = "n't";
  
  public static final Pattern PATTERN_WHITESPACE = Pattern.compile("[\\s\n\r\t]");
  
  public static final int PROGRESS_REPORT_QTY = 10000;
  
}
