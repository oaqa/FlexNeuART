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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import edu.cmu.lti.oaqa.flexneuart.utils.Const;

/**
 * A bunch of useful functions to work with XML files. In particular,
 * with XML files that store data for indexable fields. 
 * 
 * <p>Two notable XML formats:
 * <ul>
 * <li>A generic simple two-level XML (DOC -&gt; FIELD_NAME -&gt; FIELD_CONTENT);
 * <li>A more complex, more deeply nested, AQUAINT format, which we call
 * a three-level format. See {@link #parseXMLAQUAINTEntry(String)} for details.
 * </ul>
 * 
 * @author Leonid Boytsov
 *
 */
public class XmlHelper {
  public static final String AQUAINT_TEXT = "TEXT";
  public static final String AQUAINT_BODY = "BODY";
  private static final String AQUAINT_TEXT_OPEN_TAG = "<" + AQUAINT_TEXT + ">";
  private static final String AQUAINT_TEXT_CLOSE_TAG =   "</" + AQUAINT_TEXT + ">";
  private static final String AQUAINT_TEXT_SELFCLOSE_TAG = "<" + AQUAINT_TEXT + "/>";
  
  /*
   * Functions getNode and getNodeValue are 
   * based on the code from Dr. Dobbs Journal:
   * http://www.drdobbs.com/jvm/easy-dom-parsing-in-java/231002580
   */
  
  /**
   * Find node by node while ignoring the case.
   * 
   * @param tagName     node tag (case-insensitive).
   * @param nodes       a list of nodes to look through.
   * @return a node name if the node is found, or null otherwise.
   */
  public static Node getNode(String tagName, NodeList nodes) {
    for ( int x = 0; x < nodes.getLength(); x++ ) {
        Node node = nodes.item(x);
        if (node.getNodeName().equalsIgnoreCase(tagName)) {
            return node;
        }
    }
 
    return null;
  }
 
  /**
   * Get textual content of a node(for text nodes only).
   * 
   * @param node a text node.
   * @return node's textual content.
   */
  public static String getNodeValue(Node node ) {
    NodeList childNodes = node.getChildNodes();
    for (int x = 0; x < childNodes.getLength(); x++ ) {
        Node data = childNodes.item(x);
        if ( data.getNodeType() == Node.TEXT_NODE )
            return data.getTextContent();
    }
    return "";
  }

  /**
   * Generates an XML entry that can be consumed by indexing applications.
   * 
   * @param fields  (key, value) pairs; key is a field name, value is a text of the field.
   * 
   * @return an entry in a two-level pseudo-XML (namely XML without declaration) format.
   * 
   * @throws ParserConfigurationException
   * @throws TransformerException
   */
  public String genXMLIndexEntry(Map <String,String> fields) 
      throws ParserConfigurationException, 
            TransformerException {
    DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
    DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
   
    Document doc = docBuilder.newDocument();
    doc.setXmlVersion(Const.XML_VERSION);
    Element rootElement = doc.createElement(Const.TAG_DOC_ENTRY);
    doc.appendChild(rootElement);
   
    for (String key: fields.keySet()) {
      Element elemText = doc.createElement(key);
      elemText.appendChild(doc.createTextNode(fields.get(key)));
      rootElement.appendChild(elemText);
    }
    
    DOMSource source = new DOMSource(doc);
    
    
    TransformerFactory transformerFactory = TransformerFactory.newInstance();
    Transformer transformer = transformerFactory.newTransformer();
    
    StringWriter stringBuffer = new StringWriter();
    StreamResult result = new StreamResult(stringBuffer);    
    transformer.transform(source, result);
    String docStr = stringBuffer.toString();
    // Let's remove <?xml version="..." encoding="UTF-8" standalone="no"?>
    return removeHeader(docStr);
  }
  
  /**
   * 
   * Removes an XML declaration.
   * 
   * @param docStr input string
   * @return an XML without declaration.
   */
  public String removeHeader(String docStr) {
    // Let's remove <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    int indx = docStr.indexOf('>');
    // If indx == -1, indx + 1 == 0 => we output the complete string
    String docNoXmlHead = docStr.substring(indx + 1);
    return docNoXmlHead;    
  }
  
  /**
   *  Parse a standard two-level XML entry in a semi-Indri format,
   *  which was produced by the function {@link #genXMLIndexEntry(Map)}.  
   *  
   *  @param text      a textual representation of the XML entry.
   *  
   *  @return  a map, where keys are field names, while values represent
   *           values of indexable fields.
   *  @throws Exception
   */
  public static Map<String, String> parseXMLIndexEntry(String text) throws Exception {
    HashMap<String, String> res = new HashMap<String,String>();
 
    
    Document doc = parseDocWithoutXMLDecl(text);
    
    Node root = XmlHelper.getNode(Const.TAG_DOC_ENTRY, doc.getChildNodes());
    if (root == null) {
      System.err.println("Parsing error, offending document:" + Const.NL + text);
      throw new Exception("No " + Const.TAG_DOC_ENTRY);
    }  

    NodeList nodes = root.getChildNodes();
    for ( int x = 0; x < nodes.getLength(); x++ ) {
      Node node = nodes.item(x);
      
      if (node.getNodeType() == Node.ELEMENT_NODE) {
          res.put(node.getNodeName(), getNodeValue(node));
      }
    }    
    
    return res;
  }
  
  private static final 
  String CLOSING_TAG = "</"  + Const.TAG_DOC_ENTRY + ">";

  /**
   * Several entries produced produced by the function {@link #genXMLIndexEntry(Map)}
   * can be concatenated in a single file. 
   * 
   * <p>
   * This function helps read them one
   * by one. It assumes that each indexable entry <b>starts on a new line</b>,
   * that is <b>not shared</b> with any other indexable entry.
   * </p>
   * 
   * @param inpText input text
   * @return next entry, or null, if no further entry can be found.
   * @throws IOException
   */
  public static String readNextXMLIndexEntry(BufferedReader inpText) throws IOException {
    String docLine = inpText.readLine();

    if (docLine == null) return null;

    StringBuilder docBuffer = new StringBuilder();

    boolean foundEnd = false;

    do {
      docBuffer.append(docLine); docBuffer.append(Const.NL);
      if (docLine.trim().endsWith(CLOSING_TAG)) {
        foundEnd = true;
        break;        
      }
      docLine = inpText.readLine();
    } while (docLine != null);

    return foundEnd ? docBuffer.toString() : null;
  }
  
  /**
   *  Parses a more complex (two-level) entry in the AQUAINT format.
   *  We extract all 2d level fields, plus only one (TEXT) 3d level field.
   *  
   *  @param docText     a textual representation of the AQUAINT XML entry.
   *  
   *  @return  a map, where keys are field names, while values represent
   *           values of indexable fields.
   *  @throws Exception 
   */
  public Map<String, String> parseXMLAQUAINTEntry(String docText) throws Exception {
    HashMap<String, String> res = new HashMap<String,String>();
     
    TransformerFactory transformerFactory = TransformerFactory.newInstance();
    Transformer transformer = transformerFactory.newTransformer();  
    Document doc = parseDocWithoutXMLDecl(docText);
    
    Node root = XmlHelper.getNode(Const.TAG_DOC_ENTRY, doc.getChildNodes());
    if (root == null) {
      System.err.println("Parsing error, offending document:" + Const.NL + docText);
      throw new Exception("No " + Const.TAG_DOC_ENTRY);
    }  

    NodeList nodes = root.getChildNodes();
    for ( int x = 0; x < nodes.getLength(); x++ ) {
      Node node = nodes.item(x);
      
      if (node.getNodeType() == Node.ELEMENT_NODE) {
         String nodeName = node.getNodeName();
         if (!nodeName.equals(AQUAINT_BODY)) {
           res.put(nodeName, getNodeValue(node));
         } else {
           String textField = "";
           NodeList bodyNodes = node.getChildNodes();
           for ( int y = 0; y < bodyNodes.getLength(); ++ y) {
             Node bodyNode = bodyNodes.item(y);
             if (bodyNode.getNodeType() == Node.ELEMENT_NODE &&
                 bodyNode.getNodeName().equals(AQUAINT_TEXT)) {
               StringWriter stringBuffer = new StringWriter();
               StreamResult result = new StreamResult(stringBuffer);    
               transformer.transform(new DOMSource(bodyNode), result);
               
               /*
                * After getting XML, we need to remove the header AND 
                * the enclosing text tags.
                */
               textField = removeHeader(stringBuffer.toString()).trim();

               if (textField.startsWith(AQUAINT_TEXT_OPEN_TAG))
                 textField = textField.substring(AQUAINT_TEXT_OPEN_TAG.length());
               if (textField.endsWith(AQUAINT_TEXT_CLOSE_TAG))
                 textField = textField.substring(0, textField.length() - AQUAINT_TEXT_CLOSE_TAG.length());
               if (textField.equals(AQUAINT_TEXT_SELFCLOSE_TAG))
                 textField = "";
             }             
           }
           res.put(AQUAINT_TEXT, textField);
         }
      }
    }    
    
    return res;
  }
      
  /**
   * Parses an XML document that comes without an XML declaration.
   * 
   * @param docLine a textual representation of XML document.
   * @return an object of the type {@link org.w3c.dom.Document}.
   * @throws ParserConfigurationException
   * @throws SAXException
   * @throws IOException
   */
  public static Document parseDocWithoutXMLDecl(String docLine) 
      throws ParserConfigurationException, SAXException, IOException {
    String xml = String.format(
        "<?xml version=\"%s\"  encoding=\"%s\" ?>%s",
        Const.XML_VERSION, Const.ENCODING_NAME, docLine);
    return parseDocument(xml);

  }
  
  /**
   * A wrapper function to parse XML represented by a string.
   * 
   * @param docText     a textual representation of an XML document.
   * @return an object of the type {@link org.w3c.dom.Document}.
   * @throws ParserConfigurationException
   * @throws SAXException
   * @throws IOException
   */
  public static Document parseDocument(String docText) 
      throws ParserConfigurationException, SAXException, IOException {
    DocumentBuilder dbld = 
        DocumentBuilderFactory.newInstance().newDocumentBuilder();
    return dbld.parse(new InputSource(new StringReader(docText)));
  }  
  
  // Gratefully reusing the solution from StackOverflow to remove 
  // invalid XML characters: http://stackoverflow.com/a/4237934
  // with ONE DIFFERENCE: for simplicity I exclude surrogate characters.
  // For the ref. also see https://www.w3.org/TR/REC-xml/#charsets
  public static Pattern xml10pattern = Pattern.compile("[^"
      + "\u0009\r\n"
      + "\u0020-\uD7FF"
      + "\uE000-\uFFFD"
      + "]"); 
  
  public static String removeInvaildXMLChars(String s) {
    Matcher m = xml10pattern.matcher(s);
    return m.replaceAll(" ");
  }  

}
