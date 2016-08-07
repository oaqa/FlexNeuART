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
package edu.cmu.lti.oaqa.knn4qa.collection_reader;

import java.io.IOException;
import java.util.*;

import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.*;
import org.xml.sax.SAXException;

import edu.cmu.lti.oaqa.annographix.util.XmlHelper;

public class YahooAnswersParser {
  public static ParsedQuestion parse(String docText, boolean doCleanUp) throws Exception {    
    Document doc = null;
    
    try {
      doc = XmlHelper.parseDocWithoutXMLDecl(docText);
    } catch (ParserConfigurationException e1) {
      throw new IOException(e1.getMessage());
    } catch (SAXException e1) {
      throw new Exception(e1.getMessage());
    }
    
    Node docRoot  = XmlHelper.getNode("document", doc.getChildNodes());
    Node subj     = XmlHelper.getNode("subject", docRoot.getChildNodes());
    Node uriNode      = XmlHelper.getNode("uri", docRoot.getChildNodes());
    
    if (null == uriNode) {
      throw new Exception("uri node is null!");
    }
    
    String uriStr = XmlHelper.getNodeValue(uriNode);
    
    if (null == subj) {
      throw new Exception("subject node is null for uri='"+uriStr+"'!");
    }
    
    String subjText = XmlHelper.getNodeValue(subj);
    
    Node detail = XmlHelper.getNode("content", docRoot.getChildNodes());
    String questDetail = ""; 
    
    if (null != detail) {
      questDetail = XmlHelper.getNodeValue(detail);
//      if (questDetail.equals(subjText)) questDetail = "";
    }
    
    
    Node bestAnswNode = XmlHelper.getNode("bestanswer", docRoot.getChildNodes());
    
    if (null == bestAnswNode) {
      throw new Exception("bestanswer node is null for uri='"+uriStr+"'!");
    }
    
    String bestAnsw = XmlHelper.getNodeValue(bestAnswNode);
    
    ArrayList<String>   answers = new ArrayList<String>();
    
    Node answersParentNode  = XmlHelper.getNode("nbestanswers", docRoot.getChildNodes());
    
    if (null == answersParentNode) {
      throw new Exception("nbestanswers node is null for uri='"+uriStr+"'!");
    }
    
    NodeList answersNodes = answersParentNode.getChildNodes();
    
    for (int i = 0; i < answersNodes.getLength(); ++i) {
      Node item = answersNodes.item(i);
      if (item.getNodeName().equals("answer_item")) {
        String answText = XmlHelper.getNodeValue(item);
        answers.add(answText);
      }
    }
       
    return new ParsedQuestion(subjText, questDetail, uriStr, 
                              answers, bestAnsw, doCleanUp);      
  }
}
