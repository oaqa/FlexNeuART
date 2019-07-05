/*
 *  Copyright 2016 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.util.*;
import java.io.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.cli.*;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;



/**
 * Converting StackOverflow posts to Yahoo! Answers format. Tags are discarded,
 * contents of the code tags are processed in a special way: punctuation is replaced
 * with spaces, tokens smaller than {@link MIN_CODE_CHARS} characters are discarded.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertStackOverflowOnlyBest extends ConvertStackOverflowBase {
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ConvertStackOverflowOnlyBest", opt);     
    System.exit(1);
  }
  
  public static void main(String args[]) {
    
    Options options = new Options();
    
    options.addOption(INPUT_PARAM,   null, true, INPUT_DESC);
    options.addOption(OUTPUT_PARAM,  null, true, OUTPUT_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM, null, true, CommonParams.MAX_NUM_REC_DESC);
    options.addOption(DEBUG_PRINT_PARAM,   null, false, DEBUG_PRINT_DESC);
    options.addOption(EXCLUDE_CODE_PARAM,  null, false, EXCLUDE_CODE_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    HashMap<String, ParsedPost> hQuestions = new HashMap<String, ParsedPost>(); 
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String inputFile = cmd.getOptionValue(INPUT_PARAM);
      
      if (null == inputFile) Usage("Specify: " + INPUT_PARAM, options);
      
      String outputFile = cmd.getOptionValue(OUTPUT_PARAM);
      
      if (null == outputFile) Usage("Specify: " + OUTPUT_PARAM, options);
      
      InputStream input = CompressUtils.createInputStream(inputFile);
      BufferedWriter  output = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outputFile)));
      
      int maxNumRec = Integer.MAX_VALUE;
      
      String tmp = cmd.getOptionValue(CommonParams.MAX_NUM_REC_PARAM);
      
      if (tmp !=null) maxNumRec = Integer.parseInt(tmp);
      
      boolean debug = cmd.hasOption(DEBUG_PRINT_PARAM);
      
      boolean excludeCode = cmd.hasOption(EXCLUDE_CODE_PARAM);
      
      System.out.println("Processing at most " + maxNumRec + " records, excluding code? " + excludeCode);
      
      XmlIterator xi = new XmlIterator(input, ROOT_POST_TAG);
      
      String elem;
      
      output.write("<?xml version='1.0' encoding='UTF-8'?><ystfeed>\n");

      int num = 1;
      for (; num <= maxNumRec && !(elem = xi.readNext()).isEmpty(); ++num) {
        ParsedPost post = null;
        try {
          post = parsePost(elem, excludeCode);
          
          if (!post.mAcceptedAnswerId.isEmpty()) {
            hQuestions.put(post.mId, post);
          } else if (post.mpostIdType.equals("2")) {
            String parentId =  post.mParentId;
            String id = post.mId;
            if (!parentId.isEmpty()) {
              ParsedPost parentPost = hQuestions.get(parentId);
              if (parentPost != null && parentPost.mAcceptedAnswerId.equals(id)) {
                output.write(createYahooAnswersQuestion(excludeCode, parentPost, post));
                hQuestions.remove(parentId);
              }
            }
          }
          
        } catch (Exception e) {
          e.printStackTrace();
          throw new Exception("Error parsing record # " + num + ", error message: " + e);
        }
        if (debug) printDebugPost(post);
        if (num % PRINT_QTY == 0) {
          System.out.println("Processed " + num + " input recs");
        }
      }      

      output.write("</ystfeed>\n");
      
      input.close();
      output.close();
      
      System.out.println("Processed " + num + " input recs");
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }      

  }

  public static String createYahooAnswersQuestion(boolean excludeCode, ParsedPost parentPost, ParsedPost post)
      throws ParserConfigurationException, TransformerException {
    DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
    DocumentBuilder docBuilder = docFactory.newDocumentBuilder();

    // root elements
    Document doc = docBuilder.newDocument();
    Element rootElement = doc.createElement("document");

    doc.appendChild(rootElement);

    Element uri = doc.createElement("uri");
    uri.setTextContent(parentPost.mId);
    rootElement.appendChild(uri);

    Element subject = doc.createElement("subject");
    subject.setTextContent(parentPost.mTitle);
    rootElement.appendChild(subject);

    Element content = doc.createElement("content");
    content.setTextContent(parentPost.mBody);
    rootElement.appendChild(content);

    Element bestanswer = doc.createElement("bestanswer");
    bestanswer.setTextContent(post.mBody);
    rootElement.appendChild(bestanswer);

    Element answer_item = doc.createElement("answer_item");
    answer_item.setTextContent(post.mBody);
    Element nbestanswers = doc.createElement("nbestanswers");
    nbestanswers.appendChild(answer_item);
    rootElement.appendChild(nbestanswers);

    TransformerFactory transformerFactory = TransformerFactory.newInstance();
    Transformer transformer = transformerFactory.newTransformer();
    DOMSource source = new DOMSource(doc);

    StringWriter sw = new StringWriter();
    StreamResult result = new StreamResult(sw);

    transformer.transform(source, result);
    return "<vespaadd>" + xhlp.removeHeader(sw.toString()).replace("&", "&amp;") + "</vespaadd>\n";
  }  
}
