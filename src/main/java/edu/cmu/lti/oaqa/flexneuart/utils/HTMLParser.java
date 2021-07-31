package edu.cmu.lti.oaqa.flexneuart.utils;
/*
 * HTML-parser/cleaner used in TREC 19,20,21 adhoc.
 *
 * Boytsov, L., Belova, A., 2011. Evaluating Learning-to-Rank Methods in the Web Track Adhoc Task. 
 * In TREC-20: Proceedings of the Nineteenth Text REtrieval Conference.  
 *
 * Author: Leonid Boytsov
 * Copyright (c) 2013
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 * 
 */

import org.htmlparser.Parser;
import org.htmlparser.util.ParserException;

/**
 * Simple HTML Parser extracting title, meta tags, and body text
 * that is based on org.htmlparser.
 */
public class HTMLParser {

  public static HtmlDocData parse(String encoding,
                       String baseHref,
                       String html) {
    String title = "";
    String bodyText = "";
    String allText = "";
    String linkText = "";

    /*
     * 
     * This is clearly not the most efficient way to parse,
     * but it is much more stable.
     * 
     */

    try {
      Parser HtmlParser = Parser.createParser(html, encoding);  

      CleanerUtil res = new CleanerUtil(baseHref);      
      
      try {
        HtmlParser.visitAllNodesWith(res);
      } catch (StackOverflowError e) {
        System.err.println("Ouch HtmlParser has overflown the Stack!");
      }

      title = res.GetTitleText();
      bodyText = res.GetBodyText();
      linkText = res.GetLinkText();
      // res.GetDescriptionText()  Let's not use this one, it's often just spam res.GetKeywordText()

    } catch (ParserException e) {      
      System.err.println(" Parser exception: " + e + " trying simple conversion");
      // Plan B!!!
      Pair<String,String> sres = CleanerUtil.SimpleProc(html);
      
      title = sres.getFirst();
      bodyText = sres.getSecond();
      linkText = "";
    }   
    
    allText  = title + " " +  bodyText + " " + linkText;
    
    return new HtmlDocData(title, bodyText, linkText, allText);
  }

}
