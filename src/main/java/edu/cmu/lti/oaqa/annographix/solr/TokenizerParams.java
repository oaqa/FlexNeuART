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
package edu.cmu.lti.oaqa.annographix.solr;

import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.analysis.util.AbstractAnalysisFactory;

/**
 * A helper class that keeps info necessary to create a SOLR/Lucene tokenizer.
 * 
 * @author Leonid Boytsov
 *
 */
public class TokenizerParams {
  /**
   * Add a tokenizer parameter/argument.
   * 
   * @param name    a parameter name.
   * @param val     a textual representation of the parameter value.
   */
  void addArgument(String name, String val) {
    mTokClassArgs.put(name, val);
  }
  
  /**
   * @return the name of the class tokenizer factory.
   */
  public String getTokClassName() {
    return mTokClassName;
  }

  /**
   * @return the tokenizer factory class arguments.
   */
  public Map<String, String> getTokClassArgs() {
    return mTokClassArgs;
  }  
  
  /**
   * Constructor.
   * 
   * @param tokClassName    the name of the class tokenizer factory, either a short one
   *                        such as <b>solr.WhitespaceTokenizerFactory</b>
   *                        or a full class name.
   */
  TokenizerParams(String tokClassName) {
    mTokClassName = tokClassName;
    mTokClassArgs = new HashMap<String,String>();
    addLuceneVersionParam();
  }
  
  /**
   * Constructor.
   * 
   * @param tokClassName    the name of the class tokenizer factory, either a short one
   *                        such as <b>solr.WhitespaceTokenizerFactory</b>
   *                        or a full class name.
   * @param tokClassArgs    tokenizer factory class arguments, we create a new
   *                        map and copy key-value pairs their.
   */
  TokenizerParams(String tokClassName, Map<String, String> tokClassArgs) {
    mTokClassName = tokClassName;
    mTokClassArgs = new HashMap<String,String>();
    for (Map.Entry<String, String> e : tokClassArgs.entrySet()) {
      addArgument(e.getKey(), e.getValue());
    }
    addLuceneVersionParam();
  }
  

  /** Specifying which Lucene version we need */
  private void addLuceneVersionParam() {
    mTokClassArgs.put(AbstractAnalysisFactory.LUCENE_MATCH_VERSION_PARAM, UtilConst.LUCENE_VERSION);  
  }


  /** tokenizer factory class name, e.g., solr.WhitespaceTokenizerFactory */
  private String                mTokClassName;
  /** tokenizer arguments */
  private Map<String, String>   mTokClassArgs;
}