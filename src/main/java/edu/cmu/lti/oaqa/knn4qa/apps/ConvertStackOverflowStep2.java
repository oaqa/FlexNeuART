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

import static edu.cmu.lti.oaqa.knn4qa.apps.ConvertStackOverflowBase.xhlp;

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

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;


/**
 * Converting StackOverflow posts to Yahoo! Answers format: The second step. Tags are discarded,
 * contents of the code tags are processed in a special way: punctuation is replaced
 * with spaces, tokens smaller than {@link MIN_CODE_CHARS} characters are discarded.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertStackOverflowStep2 extends ConvertStackOverflowBase {
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ConvertStackOverflowOnlyStep2", opt);     
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
    
    HashSet<String>       hSeenId = new HashSet<String>();
    ArrayList<ParsedPost> postList = new ArrayList<ParsedPost>();
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String inputFile = cmd.getOptionValue(INPUT_PARAM);
      
      if (null == inputFile) Usage("Specify: " + INPUT_PARAM, options);
      
      String outputFile = cmd.getOptionValue(OUTPUT_PARAM);
      
      if (null == outputFile) Usage("Specify: " + OUTPUT_PARAM, options);
      
      BufferedReader input = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inputFile)));
      BufferedWriter  output = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(outputFile)));
      
      int maxNumRec = Integer.MAX_VALUE;
      
      String tmp = cmd.getOptionValue(CommonParams.MAX_NUM_REC_PARAM);
      
      if (tmp !=null) maxNumRec = Integer.parseInt(tmp);
      
      boolean debug = cmd.hasOption(DEBUG_PRINT_PARAM);
      
      boolean excludeCode = cmd.hasOption(EXCLUDE_CODE_PARAM);
      
      System.out.println("Processing at most " + maxNumRec + " records, excluding code? " + excludeCode);
      
      String line;
      
      output.write("<?xml version='1.0' encoding='UTF-8'?><ystfeed>\n");

      int num = 1;
      
      String prevParentId = null;
      
      for (; num <= maxNumRec && (line = input.readLine()) != null; ++num) {
        ParsedPost post = null;
        int sp1 = line.indexOf(' ');
        int sp2 = line.indexOf(' ', sp1 + 1);
        
        if (sp1 < 1 || sp2 < sp1 + 1) {
          throw new Exception("Wrong format of the prefix in line #" + num + 
              " expecting <parent id><space><post id><space>..., record:\n" + line);
        }
        String parentIdFromPrefix = line.substring(0, sp1);
        String postIdFromPrefix = line.substring(sp1+1, sp2);

        String elem = line.substring(sp2 + 1);
        
        if (prevParentId != null && !prevParentId.equals(parentIdFromPrefix)) {
          String lst = createYahooAnswersQuestion(num, output, postList);
          if (!lst.isEmpty()) { 
            output.write(lst);
            output.newLine();
          }
          
          postList.clear();
        }
        
        try {
          post = parsePost(elem, excludeCode);
          
          String postId   = post.mId;
          String parentId = post.mpostIdType.equals("2") ? post.mParentId : post.mId;
         
          // Basic check that parentIdFromPrefix and postIdFromPrefix are the same as in the post!          
          if (!postId.equals(postIdFromPrefix) || !parentId.equals(parentIdFromPrefix)) {
            throw new Exception("One of the prefix IDs don't match ID in the XML in line #" + num + 
                                " parentId (from prefix): '" + parentIdFromPrefix + "' Id (from prefix): '" + postIdFromPrefix + "' " +
                                " parentId (from post): '" + parentId + "' Id (from post): '" + postId + "'" +
                                
                                ", record:\n" + line);
          }         
          
          if (post.mpostIdType.equals("1")) {
            // if prevParentId.equals(parentId), then hSeen would have an entry with parentId
            // so that that an exception would be fired.
            if (hSeenId.contains(postId)) {
              throw new Exception("Seems like a duplicate answer post id in line #" + num + 
                                   ", record:\n" + line);              
            } else hSeenId.add(postId);
          } else if (post.mpostIdType.equals("2")) {
            if (!hSeenId.contains(parentId)) {
              throw new Exception("Question post seems to occurr before the answer in line #" + num + 
                  ", record:\n" + line);                            
            }
            if (!parentId.equals(prevParentId)) {
              throw new Exception("Answer parent ID doesn't match parent id of the previous line, in line #" + num + 
                  " parentId: '" + parentId + "' previous parent id: '" + prevParentId + "'" + 
                  ", record:\n" + line);              
            }
          } else throw new Exception("Wrong post type: " + post.mpostIdType + " in line #" + num);
          
          postList.add(post);
          prevParentId = parentIdFromPrefix;
        } catch (Exception e) {
          e.printStackTrace();
          throw new Exception("Error parsing record # " + num + ", error message: " + e);
        }
        if (debug) printDebugPost(post);
        if (num % PRINT_QTY == 0) {
          System.out.println("Processed " + num + " input recs");
        }
      }
      
      if (!postList.isEmpty()) {
        String lst = createYahooAnswersQuestion(num, output, postList);
        if (!lst.isEmpty()) { 
          output.write(lst);
          output.newLine();
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

  private static String createYahooAnswersQuestion(int lineNum, BufferedWriter output, ArrayList<ParsedPost> postList) 
                                  throws ParserConfigurationException, TransformerException {        
    if (postList.isEmpty()) throw new RuntimeException("Bug: Empty list of answers/questions! Output at line # " + lineNum);
    ParsedPost quest = postList.get(0);
    if (!quest.mpostIdType.equals("1")) 
      throw new RuntimeException("Bug: First entry in the array is not a question! Output at line # " + lineNum); 
    if (postList.size() < 2) return ""; // only a question, ignore
    int bestAnswId = -1;
    for (int i = 1; i < postList.size(); ++i) {
      ParsedPost answ = postList.get(i);
      if (!answ.mpostIdType.equals("2")) 
        throw new RuntimeException("Bug: entry " + i + " in the array is not an answer, it has type: " + answ.mpostIdType + 
                                   " id: '" + answ.mId + "'! Output at line # " + lineNum); 
      if (quest.mAcceptedAnswerId != null && quest.mAcceptedAnswerId.equals(answ.mId)) {
        bestAnswId = i; break;
      }
    }
    ParsedPost bestAnsw = null;
    if (bestAnswId >= 0) 
      bestAnsw = postList.get(bestAnswId);
    
    DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
    DocumentBuilder docBuilder = docFactory.newDocumentBuilder();

    // root elements
    Document doc = docBuilder.newDocument();
    Element rootElement = doc.createElement("document");

    doc.appendChild(rootElement);

    Element uri = doc.createElement("uri");
    uri.setTextContent(quest.mId);
    rootElement.appendChild(uri);

    Element subject = doc.createElement("subject");
    subject.setTextContent(quest.mTitle);
    rootElement.appendChild(subject);

    Element content = doc.createElement("content");
    content.setTextContent(quest.mBody);
    rootElement.appendChild(content);

    Element bestanswer = doc.createElement("bestanswer");
    if (bestAnsw != null) {      
      bestanswer.setTextContent(bestAnsw.mBody);
            
    } else {
      bestanswer.setTextContent("");
    }
    rootElement.appendChild(bestanswer);
    
    
    Element nbestanswers = doc.createElement("nbestanswers");
    for (int i = 1; i < postList.size(); ++i) {
      Element answer_item = doc.createElement("answer_item");
      answer_item.setTextContent(postList.get(i).mBody);      
      nbestanswers.appendChild(answer_item);
    }
    
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
