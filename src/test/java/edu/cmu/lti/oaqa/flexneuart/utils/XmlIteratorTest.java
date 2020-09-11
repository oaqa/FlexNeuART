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
package edu.cmu.lti.oaqa.flexneuart.utils;

import static org.junit.Assert.*;

import java.util.*;
import java.io.*;

import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.utils.XmlIterator;


/**
 * @author Leonid Boytsov
 */
public class XmlIteratorTest {
  final static Logger logger = LoggerFactory.getLogger(XmlIteratorTest.class);

  @Test
  public void test1() {
    // This string deliberately includes UTF-8 characters in Russian
    String input = "<az/>" + 
                   "<a><x>элемент 1</x></a>" +
                   "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>" +
                   "<az>some irrelevant piece</az>" + 
                   "<a attrib=\"test\" />";
    String output[] = {
        "<a><x>элемент 1</x></a>",
        "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>",
        "<a attrib=\"test\" />"  
    };
    assertTrue(check("a", input, output));
  }
  
  @Test
  public void test2() {
    // This string deliberately includes UTF-8 characters in Russian
    String input = "<az/>" + 
                   "<a ><x>элемент 1</x></a  >" +
                   "<a att1=\"test1\" att2=\"test2\"\t/>\n" +
                   "<a attrib=\"test\" />";
    String output[] = {
        "<a ><x>элемент 1</x></a  >",
        "<a att1=\"test1\" att2=\"test2\"\t/>",
        "<a attrib=\"test\" />"  
    };
    assertTrue(check("a", input, output));
  }
  
  @Test
  public void test3() {
    // This string deliberately includes UTF-8 characters in Russian
    String input = "<tagz/>" + 
                   "<tag ><x>элемент 1</x></tag  >" +
                   "<tag att1=\"test1\" att2=\"test2\"\t/>\n" +
                   "<tag attrib=\"test\" />";
    String output[] = {
        "<tag ><x>элемент 1</x></tag  >",
        "<tag att1=\"test1\" att2=\"test2\"\t/>",
        "<tag attrib=\"test\" />"  
    };
    // It's important to test tags of varying lengths
    assertTrue(check("tag", input, output));
  }
  
  @Test
  public void test4() {
    // This string deliberately includes UTF-8 characters in Russian
    // Let's also try to keep each string on a new line
    String input = "<tagz/>\n" + 
                   "<tag ><x>элемент 1</x></tag  >\n" +
                   "<tag att1=\"test1\" att2=\"test2\"\t><tagz att2=#>test again</tagz></tag><tagx/ >\n" +
                   "<tag attrib=\"test\" />\n";
    String output[] = {
        "<tag ><x>элемент 1</x></tag  >",
        "<tag att1=\"test1\" att2=\"test2\"\t><tagz att2=#>test again</tagz></tag>",
        "<tag attrib=\"test\" />"  
    };
    // It's important to test tags of varying lengths
    assertTrue(check("tag", input, output));
  }  
  
  @Test
  public void test5() {
    // Let's trying a bit more complex example
    String input = 
    "<x><some_irrelevant_tag/></x><page><title>AccessibleComputing</title>"+
    "<redirect title=\"Computer accessibility\" /><revision><id>381202555</id></revision></page>" +
    "  <page><title>Anarchism</title><ns>0</ns><id>12</id>"+
    "<revision><id>605992325</id><parentid>605973323</parentid>"+
    "<minor /><comment>[[WP:CHECKWIKI]] error fix for #61.  Punctuation goes before References. Do [[Wikipedia:GENFIXES|general fixes]] if a problem exists. - using [[Project:AWB|AWB]] (10084)</comment></page>"+
    "<page><title>AfghanistanTransportations</title><ns>0</ns><id>19</id>"+
    "<redirect title=\"Transport in Afghanistan\" />"+
    "<revision><id>409266982</id>"+
      "<parentid>74466423</parentid>"+
      "<timestamp>2011-01-22T00:37:20Z</timestamp>"+
      "<contributor>"+
        "<username>Asklepiades</username>"+
        "<id>930338</id>"+
      "</contributor>"+
      "<text xml:space=\"preserve\">#REDIRECT [[Transport in Afghanistan]] {{R from CamelCase}} {{R unprintworthy}}</text>"+
      "<sha1>lx95oyrvksg2uiro2uya214r154onuz</sha1>"+
      "<model>wikitext</model>"+
      "<format>text/x-wiki</format>"+
    "</revision>"+
    "</page>"    
    
    ;
        
    String output[] = {
    "<page><title>AccessibleComputing</title>"+
    "<redirect title=\"Computer accessibility\" /><revision><id>381202555</id></revision></page>",
    "<page><title>Anarchism</title><ns>0</ns><id>12</id>"+
    "<revision><id>605992325</id><parentid>605973323</parentid>"+
    "<minor /><comment>[[WP:CHECKWIKI]] error fix for #61.  Punctuation goes before References. Do [[Wikipedia:GENFIXES|general fixes]] if a problem exists. - using [[Project:AWB|AWB]] (10084)</comment></page>",
    "<page><title>AfghanistanTransportations</title><ns>0</ns><id>19</id>"+
    "<redirect title=\"Transport in Afghanistan\" />"+
    "<revision><id>409266982</id>"+
      "<parentid>74466423</parentid>"+
      "<timestamp>2011-01-22T00:37:20Z</timestamp>"+
      "<contributor>"+
        "<username>Asklepiades</username>"+
        "<id>930338</id>"+
      "</contributor>"+
      "<text xml:space=\"preserve\">#REDIRECT [[Transport in Afghanistan]] {{R from CamelCase}} {{R unprintworthy}}</text>"+
      "<sha1>lx95oyrvksg2uiro2uya214r154onuz</sha1>"+
      "<model>wikitext</model>"+
      "<format>text/x-wiki</format>"+
    "</revision>"+
    "</page>"    
    };
    // It's important to test tags of varying lengths
    assertTrue(check("page", input, output));
  }
  
  @Test
  public void test6() {
    // This test string deliberately includes UTF-8 characters in Russian
    // Let's test CDATA outside of the 'tag' (should be ignored) and inside the tag
    String input = "<tagz/><![CDATA[<tag>the tag inside the to-be-ignored CDATA-block</tag>]]>" + 
   "<tag ><x>элемент 1</x><![CDATA[<tag>the tag inside CDATA</tag>]]></tag  >" +
   "<tag ><x>элемент 1</x><![CDATA[<tag>unclosed tag inside CDATA<tag>]]><x>элемент 2</x><![CDATA[<]]></tag  >" +
   "<![CDATA[<tag>the tag inside the to-be-again-ignored CDATA-block</tag><<<<<tag>]]>"+
   "<tag ><![CDATA[<<<<<]]></tag>" +
   "<tagz><![CDATA[<tag>the tag inside the to-be-again-ignored CDATA-block</tag><<<<<tag>]]><tag>test</tag></tagz>";
    String output[] = {
        "<tag ><x>элемент 1</x><![CDATA[<tag>the tag inside CDATA</tag>]]></tag  >",
        "<tag ><x>элемент 1</x><![CDATA[<tag>unclosed tag inside CDATA<tag>]]><x>элемент 2</x><![CDATA[<]]></tag  >",
        "<tag ><![CDATA[<<<<<]]></tag>",
        "<tag>test</tag>"
    };
    // It's important to test tags of varying lengths
    assertTrue(check("tag", input, output));
  }  


  private static boolean check(String tagName, String input, String[] output) {
    InputStream is = new ByteArrayInputStream(input.getBytes());
    try {
      XmlIterator it = new XmlIterator(is, tagName);
      
      ArrayList<String> res = new ArrayList<String>();
      
      for (String s;;) {
        s = it.readNext();
        if (s.isEmpty()) break;
        res.add(s);
      }
      
      if (res.size() != output.length) {
        logger.error(String.format("Unexpected # of results, expected %d, got %d",
                                   output.length, res.size()));
        return false;
      }
      for (int i = 0; i < res.size(); ++i) {
        if (!res.get(i).equals(output[i])) {
          logger.error(String.format("Wrong element index %d, expected '%s', got '%s'",
                              i, output[i], res.get(i)));
          return false;          
        }
      }
      
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      return false;
    }
    
    return true;
  }

  @Test
  public void test7() {
    // This string deliberately includes UTF-8 characters in Russian
    String input = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"+
                   "<!DOCTYPE note SYSTEM \"Note.dtd\">" +  
                   "<az/>" + 
                   "<a><x>элемент 1</x></a>" +
                   "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>" +
                   "<az>some irrelevant piece</az>" + 
                   "<a attrib=\"test\" />";
    String output[] = {
        "<a><x>элемент 1</x></a>",
        "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>",
        "<a attrib=\"test\" />"  
    };
    assertTrue(check("a", input, output));
  }


  @Test
  public void test8() {
    // This string deliberately includes UTF-8 characters in Russian
    String input = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"+
                    // We dont't care how many times preamble appears, simply
                    // ignore it in all cases ...
                   "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"+
                    // ... also ignore DOCTYPEs
                   "<!DOCTYPE note SYSTEM \"Note.dtd\">" +                    
                   "<az/>" + 
                   "<a><x>элемент 1</x></a>" +
                   // ... also ignore *repeating* DOCTYPEs
                   "<!DOCTYPE note"+
                   "["+
                   "<!ELEMENT note (to,from,heading,body)>"+
                   "<!ELEMENT to (#PCDATA)>"+
                   "<!ELEMENT from (#PCDATA)>"+
                   "<!ELEMENT heading (#PCDATA)>"+
                   "<!ELEMENT body (#PCDATA)>"+
                   "]>"+                   
                   "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>" +
                   "<az>some irrelevant piece</az>" + 
                   "<a attrib=\"test\" />";
    String output[] = {
        "<a><x>элемент 1</x></a>",
        "<a attrib=\"test\"><y>элемент 2</y><ab>some irrelevant tag</ab></a>",
        "<a attrib=\"test\" />"  
    };
    assertTrue(check("a", input, output));
  }  
  
}
