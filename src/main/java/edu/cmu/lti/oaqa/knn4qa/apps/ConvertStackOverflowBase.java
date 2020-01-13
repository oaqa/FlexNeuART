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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


import org.htmlparser.Parser;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.util.ParserException;
import org.htmlparser.util.Translate;
import org.htmlparser.visitors.NodeVisitor;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import edu.cmu.lti.oaqa.knn4qa.utils.XmlHelper;

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


public class ConvertStackOverflowBase {
  public static final String ROOT_POST_TAG = "row";
  public static final int MIN_CODE_CHARS=3;
  public static final int PRINT_QTY = 10000;
  
  public static final String INPUT_PARAM = "input";
  public static final String INPUT_DESC  = "input file (the bzipped, gzipped, or uncompressed posts file from Stack Overflow)";
  
  public static final String OUTPUT_PARAM = "output";
  public static final String OUTPUT_DESC  = "output file (in the Yahoo! answers format";

  public static final String DEBUG_PRINT_PARAM  = "debug_print";
  public static final String DEBUG_PRINT_DESC   = "Print some debug info";
  
  public static final String EXCLUDE_CODE_PARAM = "exclude_code";
  public static final String EXCLUDE_CODE_DESC  = "Completely remove all the code sections";  
  
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
  
  
  public static void printDebugPost(ParsedPost post) {
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
    
  protected static XmlHelper xhlp = new XmlHelper();
  
}
