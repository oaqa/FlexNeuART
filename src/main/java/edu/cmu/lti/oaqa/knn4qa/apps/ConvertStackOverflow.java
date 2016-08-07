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


class ParsedPost {
  String mId;
  String mAcceptedAnswerId;
  String mParentId;
  String mpostIdType;
  String mTitle;
  String mBody;
  
  private String emptyIfNull(String s) {
    return s == null ? "" : s;
  }
  
  public ParsedPost(String id, String acceptedAnswerId, String parentId, String postIdtype, 
                    String title, String body) {    
    this.mId = id;
    this.mAcceptedAnswerId = emptyIfNull(acceptedAnswerId);
    this.mParentId = emptyIfNull(parentId);
    this.mpostIdType = postIdtype;
    this.mTitle = title;
    this.mBody = body;
  }  
};

class PostCleanerVisitor extends NodeVisitor {
    
  public PostCleanerVisitor(int minCodeChars, boolean excludeCode) {
    mMinCodeChars = minCodeChars;
    mExcludeCode = excludeCode;
    initParaTags();
  }
  
  public String getText() {
    return collapseSpaces(mTextBuffer.toString());
  }
  
  @Override
  public void visitTag(Tag tag) {
    String lcName = tag.getTagName().toLowerCase();
    
    setFlags(lcName, true);
    processParagraphs(lcName, true);
  }
  
  @Override
  public void visitEndTag(Tag tag) {
    String Name = tag.getTagName().toLowerCase();
      
    setFlags(Name, false);
    processParagraphs(Name, false);
  }  
  
  void setFlags(String lcName, boolean flag) {
    if (lcName.equals("code")) mIsInCode = flag;
  }
  
  private void processParagraphs(String lcName, boolean isStart) {
    if (mParaTags.contains(lcName)) {
      mTextBuffer.append("\n");
    }
  }
  
  public void visitStringNode(Text TextNode) {
    String text = cleanText(TextNode.getText());
    
    if (mIsInCode) {
      if (mExcludeCode) text = "";
      else {
        text = mPunctCleanUp.matcher(text).replaceAll(" ");
        ArrayList<String> parts = new ArrayList<String>();
        text = text.replaceAll("\\s", " ");
        for (String s: mSpaceSplit.split(text)) {
          if (s.length() >= mMinCodeChars) parts.add(s);
        }
        text = mSpaceJoin.join(parts);
      }
    }    

    mTextBuffer.append(text);
  }   
  
  private static String replaceNonBreakingSpaceWithOrdinarySpace(String text) {
    return text.replace('\u00a0',' ');
  }    
  
  private static String cleanText(String text) {
    return replaceNonBreakingSpaceWithOrdinarySpace(Translate.decode(text));
  }
    
  static public String collapseSpaces(String s) { 
    s = replace(s, "\\r", "\n");
    // Looks like quantifiers +, * here may lead to an infinite recursion
    s = replace(s, "[\\n]{1,10}", "\n");
    return replace(s, "[\\t ]{1,10}", " ");   
  }  
  
  private void initParaTags() {
    mParaTags = new HashSet<String>();
    // Paragraph elements
    mParaTags.add("p");
    mParaTags.add("div");
    mParaTags.add("br");
    mParaTags.add("hr");
    // Misc
    mParaTags.add("a");
    mParaTags.add("pre");
    mParaTags.add("blockquote");
    // List elements
    mParaTags.add("ol");
    mParaTags.add("ul");
    mParaTags.add("li");
    // Table elements
    mParaTags.add("tr");
    mParaTags.add("td");
    mParaTags.add("table");
    mParaTags.add("tbody");
    mParaTags.add("thead");
    mParaTags.add("tfoot");
    mParaTags.add("th");
  }  
  
  private static String replace(String src, String pat, String repl, boolean bIgnoreCase) {
    Pattern p = bIgnoreCase ? Pattern.compile(pat,  Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL) : 
                  Pattern.compile(pat,  Pattern.MULTILINE | Pattern.DOTALL);
    Matcher m = p.matcher(src);
        
    return m.replaceAll(repl);
  }
  
  // Case insensitive searching.
  private static String replace(String src, String pat, String repl) {
    return replace(src, pat, repl, true);
  }
  
  public static String simpleProc(String text) {
    text= cleanText(text); 

    text = replace(text, "<script[^>]*>.*?<[/]script>", "", true); // Remove all script tags
    text = replace(text, "<style[^>]*>.*?<[/]style>", "", true); // Remove all content inside style
    text = replace(text, "<!\\-\\-.*?\\-\\->", "");
    text = replace(text, "<[^>]*?>", " "); // Remove all tags

    return text;
  }  
  
  private boolean         mIsInCode = false;
  private StringBuffer    mTextBuffer = new StringBuffer();
  private HashSet<String> mParaTags = new HashSet<String>();
  
  private static Pattern  mPunctCleanUp = Pattern.compile("(\\p{P}|[<>()\\[\\]@~&|+-=$/])", Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);
  private static Joiner   mSpaceJoin  = Joiner.on(' ');
  private static Splitter mSpaceSplit = Splitter.on(' ').omitEmptyStrings().trimResults();
  
  private int             mMinCodeChars = 0;
  private boolean         mExcludeCode = false;
}

class PostCleaner {
  public PostCleaner(String html, int minCodeChars, boolean excludeCode) {
    try {
      Parser htmlParser = Parser.createParser(html, "utf8");  
  
      PostCleanerVisitor res = new PostCleanerVisitor(minCodeChars, excludeCode);      
      htmlParser.visitAllNodesWith(res);      
      mText = res.getText();
    } catch (ParserException e) {      
      System.err.println(" Parser exception: " + e + " trying simple conversion");
      // Plan B!!!
      mText = PostCleanerVisitor.simpleProc(html);
    }    
  }
  
  public String getText() { return mText; }
  
  private String mText;
}

/**
 * Converting StackOverflow posts to Yahoo! Answers format. Tags are discarded,
 * contents of the code tags are processed in a special way: punctuation is replaced
 * with spaces, tokens smaller than {@link MIN_CODE_CHARS} characters are discarded.
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertStackOverflow {
  private static final String ROOT_POST_TAG = "row";

  public static final int MIN_CODE_CHARS=3;

  public static final String INPUT_PARAM = "input";
  public static final String INPUT_DESC  = "input file (the bzipped, gzipped, or uncompressed posts file from Stack Overflow)";
  
  public static final String OUTPUT_PARAM = "output";
  public static final String OUTPUT_DESC  = "output file (in the Yahoo! answers format";
  
  public static final String DEBUG_PRINT_PARAM  = "debug_print";
  public static final String DEBUG_PRINT_DESC   = "Print some debug info";
  
  public static final String EXCLUDE_CODE_PARAM = "exclude_code";
  public static final String EXCLUDE_CODE_DESC  = "Completely remove all the code sections";
  
  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ConvertStackOverflow", opt);     
    System.exit(1);
  }
  
  public static ParsedPost parsePost(String postText, boolean excludeCode) throws Exception {
    Document post = XmlHelper.parseDocWithoutXMLDecl(postText);
    
    NamedNodeMap attr = post.getDocumentElement().getAttributes();
    
    if (null == attr) {
      throw new Exception("Invalid entry, no attributes!");
    }        
    
    Node  itemId = attr.getNamedItem("Id");
    Node  itemAcceptedAnswerId = attr.getNamedItem("AcceptedAnswerId");
    Node  itemParentId = attr.getNamedItem("ParentId");
    Node  itemPostTypeId = attr.getNamedItem("PostTypeId");
    Node  itemTitle= attr.getNamedItem("Title");
    Node  itemBody = attr.getNamedItem("Body");
    
    if (null == itemId)         throw new Exception("Missing Id");
    if (null == itemPostTypeId) throw new Exception("Missing PostTypeId");
    if (null == itemBody)       throw new Exception("Missing Body");
    
    String id               = XmlHelper.getNodeValue(itemId); 
    String acceptedAnswerId = itemAcceptedAnswerId != null ? XmlHelper.getNodeValue(itemAcceptedAnswerId) : "";   
    String postIdType       = XmlHelper.getNodeValue(itemPostTypeId);     
    String parentId         = itemParentId != null ? XmlHelper.getNodeValue(itemParentId) : "";
    String title            = itemTitle != null ? XmlHelper.getNodeValue(itemTitle) : "";
    String body             = XmlHelper.getNodeValue(itemBody);
    
    
    return new ParsedPost(id, acceptedAnswerId, parentId, postIdType, 
                          (new PostCleaner(title, MIN_CODE_CHARS, true)).getText(),
                          (new PostCleaner(body, MIN_CODE_CHARS, excludeCode)).getText());
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
      BufferedWriter  output = new BufferedWriter(new FileWriter(new File(outputFile)));
      
      int maxNumRec = Integer.MAX_VALUE;
      
      String tmp = cmd.getOptionValue(CommonParams.MAX_NUM_REC_PARAM);
      
      if (tmp !=null) maxNumRec = Integer.parseInt(tmp);
      
      boolean debug = cmd.hasOption(DEBUG_PRINT_PARAM);
      
      boolean excludeCode = cmd.hasOption(EXCLUDE_CODE_PARAM);
      
      System.out.println("Processing at most " + maxNumRec + " records, excluding code? " + excludeCode);
      
      XmlIterator xi = new XmlIterator(input, ROOT_POST_TAG);
      
      String elem;
      
      output.write("<?xml version='1.0' encoding='UTF-8'?><ystfeed>\n");

      for (int num = 1; num <= maxNumRec && !(elem = xi.readNext()).isEmpty(); ++num) {
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
                output.write(createYahooAnswersQuestion(parentPost, post));
                hQuestions.remove(parentId);
              }
            }
          }
          
        } catch (Exception e) {
          e.printStackTrace();
          throw new Exception("Error parsing record # " + num + ", error message: " + e);
        }
        if (debug) {
          System.out.println(String.format("%s parentId=%s acceptedAnswerId=%s type=%s", 
                             post.mId, post.mParentId, post.mAcceptedAnswerId, post.mpostIdType));
          System.out.println("================================");
          if (!post.mTitle.isEmpty()) {
            System.out.println(post.mTitle);
            System.out.println("--------------------------------");
          }
          System.out.println(post.mBody);
          System.out.println("================================");
        }
      }      

      output.write("</ystfeed>\n");
      
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

  private static String createYahooAnswersQuestion(ParsedPost parentPost, ParsedPost post) 
      throws ParserConfigurationException, TransformerException {
    DocumentBuilderFactory    docFactory = DocumentBuilderFactory.newInstance();
    DocumentBuilder           docBuilder = docFactory.newDocumentBuilder();
    
    // root elements
    Document doc = docBuilder.newDocument();
    Element  rootElement = doc.createElement("document");
    
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
    
    Element answer_item  = doc.createElement("answer_item");
    answer_item.setTextContent(post.mBody);
    Element nbestanswers = doc.createElement("nbestanswers");
    nbestanswers.appendChild(answer_item);
    rootElement.appendChild(nbestanswers);
    
    TransformerFactory transformerFactory = TransformerFactory.newInstance();
    Transformer        transformer = transformerFactory.newTransformer();
    DOMSource          source = new DOMSource(doc);
    
    
    StringWriter sw = new StringWriter();
    StreamResult result = new StreamResult(sw);
        
    transformer.transform(source, result);
    return "<vespaadd>" + xhlp.removeHeader(sw.toString()).replace("&", "&amp;") + "</vespaadd>\n";
  }
  
  private static XmlHelper xhlp = new XmlHelper();
}
