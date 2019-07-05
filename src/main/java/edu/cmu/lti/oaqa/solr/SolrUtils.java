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
package edu.cmu.lti.oaqa.solr;

import java.util.*;

import org.apache.solr.common.*;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;


import edu.cmu.lti.oaqa.knn4qa.utils.HttpHelper;


/**
 * @author Leonid Boytsov using some of the SolrUtils code written
 *                        by Elmer Garduno and Alkesh Patel.  
 *
 */
public class SolrUtils {
  private static final String FIELD_TYPE = "fieldType";

  /**
   * Take key-value pairs (field name, field text) and create an indexable document,
   * which is an object of the type {@link org.apache.solr.common.SolrInputDocument}.
   * 
   * @param    keyValueMap  key-value map (field name, field text).
   * 
   * @return an object of the type {@link org.apache.solr.common.SolrInputDocument},
   *         which can be indexed.
   * @throws Exception
   */
  public static SolrInputDocument buildSolrDocument(HashMap<String, Object> keyValueMap)
      throws Exception {

    SolrInputDocument doc = new SolrInputDocument();

    Iterator<String> keys = keyValueMap.keySet().iterator();
    while (keys.hasNext()) {
      String key = keys.next();
      Object value = keyValueMap.get(key);

      SolrInputField field = new SolrInputField(key);
      try {

        doc.addField(field.getName(), value);

      } catch (Exception e) {
        e.printStackTrace();
      }
    }

    return doc;

  }

 /**
  * Given a SOLR-server URI, retrieve the respective 
  * configuration file (solrconfig.xml).
  * 
  * @param solrURI  SOLR-server URI
  * @return configuration file contents
  * @throws Exception
  */
  public static String getSolrConfig(String solrURI) throws Exception {
    return getSolrFileContent(solrURI, "solrconfig.xml");
  }

  /**
   * Given a SOLR-server URI, retrieve the respective 
   * schema file (schema.xml).
   * 
   * @param solrURI  SOLR-server URI
   * @return configuration file contents
   * @throws Exception
   */   
  public static String getSolrSchema(String solrURI) throws Exception {
    return getSolrFileContent(solrURI, "schema.xml");
  }
  
  /**
   * Given a SOLR-server URI, retrieve the respective 
   * stopword file (stopwords.txt).
   * 
   * @param solrURI  SOLR-server URI
   * @return configuration file contents
   * @throws Exception
   */   
  public static String getSolrStopwords(String solrURI) throws Exception {
    return getSolrFileContent(solrURI, "stopwords.txt");
  }    
  
  private static String getSolrFileContent(String solrURI, String fileName) 
                        throws Exception {
    String getSchemaURI = solrURI 
        + "/admin/file?file=" + fileName + "&contentType=text/xml:charset=utf-8";
        
    return HttpHelper.get(getSchemaURI);
  }
  
  /**
   * The function checks the following: (1) there are no omit* attributes except
   * from those that match a key from the provided key-value pair map ; 
   * (2) The key-value pair map defines mandatory attributes and their values.
   * 
   * @param fieldName        a name of the field.
   * @param fieldNode        an XML node representing the field.
   * @param mandAttrKeyVal   a key-value pair map of mandatory attributes.
   * 
   * @throws                 Exception
   */
  private static void checkFieldAttrs(String fieldName, Node fieldNode, 
                                      HashMap<String, String> mandAttrKeyVal) 
    throws Exception {
    HashMap<String, String> attKeyVal = new HashMap<String, String>();
    
    NamedNodeMap attrs = fieldNode.getAttributes();
    
    if (null == attrs) {
      if (!mandAttrKeyVal.isEmpty()) 
        throw new Exception("Field: " + fieldNode.getLocalName() + 
            " should have attributes");
      return;
    }
          
    // All omit* attributes are disallowed unless they are explicitly allowed
    for (int i = 0; i < attrs.getLength(); ++i) {
      Node e = attrs.item(i);
      String nm = e.getNodeName();
      attKeyVal.put(nm, e.getNodeValue());
      if (nm.startsWith("omit") && !mandAttrKeyVal.containsKey(nm)) {
        throw new Exception("Unexpected attribute, field: '" + fieldName + "' " +  
            " shouldn't have the attribute '" + nm + "'");
      }
    }
    for (Map.Entry<String, String> e: mandAttrKeyVal.entrySet()) {
      String key = e.getKey();
      if (!attKeyVal.containsKey(key)) {
        throw new Exception("Missing attribute, field: " + fieldName  + "' " +  
            " should have an attribute '" + key  + 
            "' value: '" + e.getValue() + "'");
      }
      String expVal = e.getValue();
      String val = attKeyVal.get(key);
      if (val.compareToIgnoreCase(expVal) != 0) {
        throw new Exception("Wrong attribute value, field: '" + fieldName + "' " +  
            "attribute '" + key + "' should have the value '"+expVal+"'");                
      }
    }
  }

  private final static String NL = System.getProperty("line.separator");
}
