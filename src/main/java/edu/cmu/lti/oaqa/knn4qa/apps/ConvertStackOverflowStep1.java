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
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.cli.*;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.utils.XmlIterator;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import org.htmlparser.Parser;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.util.ParserException;
import org.htmlparser.util.Translate;
import org.htmlparser.visitors.NodeVisitor;
import org.w3c.dom.*;


/**
 * Converting StackOverflow posts to Yahoo! Answers format: The first step. We
 * simply read the data remove '\n\r' from each post/question entry. Then, we prepend
 * the following two-space separated numbers:
 * <parent id> <post id>
 * For questions <parent id>==<post id>.
 * The output of this utility will be fed to Unix utility sort.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertStackOverflowStep1 extends ConvertStackOverflowBase {
  private static final String ROOT_POST_TAG = "row";

   
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ConvertStackOverflow", opt);     
    System.exit(1);
  }
  
  
  public static void main(String args[]) {
    
    Options options = new Options();
    
    options.addOption(INPUT_PARAM,   null, true, INPUT_DESC);
    options.addOption(OUTPUT_PARAM,  null, true, OUTPUT_DESC);
    options.addOption(CommonParams.MAX_NUM_REC_PARAM, null, true, CommonParams.MAX_NUM_REC_DESC);
    options.addOption(DEBUG_PRINT_PARAM,   null, false, DEBUG_PRINT_DESC);
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    HashSet<String> hGoodQuest = new HashSet<String>(); 
    
    try {
      CommandLine cmd = parser.parse(options, args);
      
      String inputFile = cmd.getOptionValue(INPUT_PARAM);
      
      if (null == inputFile) Usage("Specify: " + INPUT_PARAM, options);
      
      String outputFile = cmd.getOptionValue(OUTPUT_PARAM);
      
      if (null == outputFile) Usage("Specify: " + OUTPUT_PARAM, options);
      
      InputStream input = CompressUtils.createInputStream(inputFile);
      BufferedWriter  output = new BufferedWriter(new FileWriter(new File(outputFile)));
      
      int maxNumRec = Integer.MAX_VALUE;
      
      String tmp = cmd.getOptionValue(CommonParams.MAX_NUM_REC_PARAM);
      
      if (tmp !=null) maxNumRec = Integer.parseInt(tmp);
      
      boolean debug = cmd.hasOption(DEBUG_PRINT_PARAM);
      
      boolean excludeCode = cmd.hasOption(EXCLUDE_CODE_PARAM);
      
      System.out.println("Processing at most " + maxNumRec + " records, excluding code? " + excludeCode);
      
      XmlIterator xi = new XmlIterator(input, ROOT_POST_TAG);
      
      String elem;

      for (int num = 1; num <= maxNumRec && !(elem = xi.readNext()).isEmpty(); ++num) {
        ParsedPost post = null;
        try {
          elem = elem.replace('\n', ' '); // Make sure that there's not newline, so we can use sort
          post = parsePost(elem, excludeCode);
          
          boolean isGood = false;
          String id = post.mId;
          
          if (!post.mAcceptedAnswerId.isEmpty()) {
            hGoodQuest.add(id);
            isGood = true;
          } else if (post.mpostIdType.equals("2")) {
            String parentId = post.mParentId;
            isGood = (!parentId.isEmpty()) && hGoodQuest.contains(parentId);
          }
          
          if (isGood) {
            String parentId = post.mpostIdType.equals("2") ? post.mParentId : post.mId;
            output.write(parentId + " " + id + " " + elem);
            output.newLine();
          }                              
        } catch (Exception e) {
          e.printStackTrace();
          throw new Exception("Error parsing record # " + num + ", error message: " + e);
        }
        if (debug) printDebugPost(post);
      }      
      
      input.close();
      output.close();
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }      

  }
}
