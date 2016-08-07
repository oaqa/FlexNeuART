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

import java.util.*;

import org.apache.solr.common.*;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.xml.sax.SAXException;

import edu.cmu.lti.oaqa.annographix.util.HttpHelper;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;

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

        doc.addField(field.getName(), value, 1.0f);

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
   * This function does two things extract tokenizer information for the main
   * indexable field and performs a config sanity check.
   * 
   * <p>
   * We need to ensure that 1) the field that stores annotations 
   * uses the whitespace tokenizer; 2) the annotated text field stores both
   * offsets and positions.
   * </p> 
   * <p>Ideally, one should be able to parse the config using standard
   * SOLR class: org.apache.solr.schema.IndexSchema.
   * However, this seems to be impossible, because the schema loader tries
   * to read/parse all the files (e.g., a stopword file) mentioned in the schema 
   * definition. These files are not accessible, though, 
   * because they clearly "sit" on a SOLR server file system.
   * </p> 
   * 
   * 
   * @param solrURI         a URI of the SOLR instance.
   * @param textFieldName   a name of the text field to be annotated.
   * @param annotFieldName  a name of the field to store annotations.
   * 
   * @return a key-value map, where keys are field names and objects
   *         are elements of the type {@link TokenizerParams}; these
   *         objects keep information about fields' tokenizers.
   * 
   * @throws Exception
   */
  public static Map<String,TokenizerParams> parseAndCheckConfig(String solrURI, 
                                                  String textFieldName,
                                                  String annotFieldName) 
                                                      throws Exception {    
    String respText = SolrUtils.getSolrSchema(solrURI);
        
    Map<String,TokenizerParams> tmpRes = new HashMap<String,TokenizerParams>();
    Map<String,TokenizerParams> res = new HashMap<String,TokenizerParams>();
    
    try {
      Document doc = XmlHelper.parseDocument(respText);
      
      Node root     = XmlHelper.getNode("schema", doc.getChildNodes());
      Node fields   = XmlHelper.getNode("fields", root.getChildNodes());
      Node types    = XmlHelper.getNode("types",  root.getChildNodes());
      
      // Let's read which tokenizers are specified for declared field types  
      HashMap<String, String> typeTokenizers = new HashMap<String, String>();
      for (int i = 0; i < types.getChildNodes().getLength(); ++i) {
        Node oneType = types.getChildNodes().item(i);
        
        if (oneType.getAttributes() == null) continue;
        if (!oneType.getNodeName().equalsIgnoreCase(FIELD_TYPE)) continue;
        Node nameAttr = oneType.getAttributes().getNamedItem("name");
        if (nameAttr != null) {
          String name = nameAttr.getNodeValue();
          
          Node tmp = XmlHelper.getNode("analyzer", oneType.getChildNodes());
          if (tmp != null) {
            tmp = XmlHelper.getNode("tokenizer", tmp.getChildNodes());
            if (tmp != null) {
              NamedNodeMap attrs = tmp.getAttributes();
              if (null == attrs)
                throw new Exception("No attributes found for the tokenizer description, " + 
                                    " type: " + name);  
              String tokClassName = null;
              Node tokAttr = attrs.getNamedItem("class");
              if (tokAttr != null) {
                tokClassName = tokAttr.getNodeValue();
              } else throw new Exception("No class specified for the tokenizer description, " + 
                  " type: " + name);
              
              typeTokenizers.put(name, tokClassName);
              TokenizerParams tokDesc = new TokenizerParams(tokClassName);              
              
              for (int k = 0; k < attrs.getLength(); ++k) {
                Node oneAttr = attrs.item(k);
                if (!oneAttr.getNodeName().equalsIgnoreCase("class")) {
                  tokDesc.addArgument(oneAttr.getNodeName(), oneAttr.getNodeValue());
                }
              }                 
              tmpRes.put(name, tokDesc);
            } else {
              tmpRes.put(name, null);
            }
          }       
        }
      }

      // Read a list of fields, check if they are configured properly
      boolean annotFieldPresent = false;
      boolean text4AnnotFieldPresent = false;
      
      for (int i = 0; i < fields.getChildNodes().getLength(); ++i) {
        Node oneField = fields.getChildNodes().item(i);
        
        if (oneField.getAttributes()==null) continue;
        Node nameAttr = oneField.getAttributes().getNamedItem("name");
        if (nameAttr == null) {
          continue;
        }
        
        String fieldName = nameAttr.getNodeValue();
        
        Node typeAttr = oneField.getAttributes().getNamedItem("type");
        
        if (null == typeAttr) {
          throw new Exception("Missing type for the annotation field: " 
              + fieldName);          
        }

        

        String val = typeAttr.getNodeValue();
        String className = typeTokenizers.get(val);
        
        res.put(fieldName, tmpRes.get(val));

        // This filed must be present, use the whitespace tokenizer, and index positions      
        if (fieldName.equalsIgnoreCase(annotFieldName)) {
          HashMap<String,String> attrVals = new HashMap<String,String>();
          attrVals.put("omitPositions", "false");
          checkFieldAttrs(fieldName, oneField, attrVals);
          annotFieldPresent = true;

          if (className == null) {
            throw new Exception("Cannot find the tokenizer class for the field: '" 
                           + fieldName + "' " + 
                          "you should explicitly specify one in the section '" +
                           FIELD_TYPE + "'!");
          }          
                    
          if (!className.equals(UtilConst.ANNOT_FIELD_TOKENIZER)) {
            throw new Exception("The field: '" + annotFieldName + "' " +
                                " should be configured to use the tokenizer: " +
                                UtilConst.ANNOT_FIELD_TOKENIZER);
          }
          
        } else if (fieldName.equalsIgnoreCase(textFieldName)) {
        /*
         *  This field must be present, and index positions as well as offsets,
         *  there should be a tokenizer specified explicitly!
         */
          text4AnnotFieldPresent = true;
          HashMap<String,String> attrVals = new HashMap<String,String>();
          attrVals.put("omitPositions", "false");
          attrVals.put("storeOffsetsWithPositions", "true");
          checkFieldAttrs(fieldName, oneField, attrVals);
          
          if (className == null) {
            throw new Exception("Cannot find the tokenizer class for the field: '" 
                           + fieldName + "' " + 
                          "you should explicitly specify one in the section '" +
                           FIELD_TYPE + "'!");
          }          
        }
      }
      if (!annotFieldPresent) {
        throw new Exception("Missing field: " + annotFieldName);
      }
      if (!text4AnnotFieldPresent) {
        throw new Exception("Missing field: " + textFieldName);
      }           
    } catch (SAXException e) {      
      System.err.println("Can't parse SOLR response:" + NL + respText);
      throw e;
    }
    
    return res;
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

  /**
   * A simple test function to check some util functions manually.
   * 
   * @param args    args[0] is a server URI, e.g., 
   *                "http://localhost:8984/solr/AQUAINT"
   * @throws Exception
   */
  public static void main(String args[]) throws Exception {
    String uri = args[0];
    System.out.println(getSolrSchema(uri));
    System.out.println("===============");
    System.out.println(getSolrConfig(uri));
    System.out.println("===============");
    System.out.println(getSolrStopwords(uri));
    System.out.println("===============");
    Map<String, TokenizerParams> res = parseAndCheckConfig(uri, 
        UtilConst.DEFAULT_TEXT4ANNOT_FIELD, UtilConst.DEFAULT_ANNOT_FIELD);
    for (Map.Entry<String, TokenizerParams> e : res.entrySet()) {
      String txt = e.getValue() != null ? e.getValue().getTokClassName() : "NULL";
      System.out.println(e.getKey() + " -> " + txt);
    }
  }

  private final static String NL = System.getProperty("line.separator");
}
