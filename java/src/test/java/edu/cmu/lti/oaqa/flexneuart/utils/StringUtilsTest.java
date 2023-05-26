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

import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Test;

public class StringUtilsTest {
  
  @Test 
  public void testRemoveLuceneOps() {
    String src[] = {
        "this is an or test AND", 
        "this is an or test and", 
        "AND OR TO NOT",
        "and or to not AND OR TO NOT"
    };
    String dst[] = {
        "this is an or test", 
        "this is an or test and", 
        "",
        "and or to not"        
    };
    for (int i = 0; i < src.length; i++) {
      assertEquals(dst[i], StringUtils.removeLuceneSpecialOps(src[i]));
    }
  }

  @Test
  public void testTruncAt() {
    ArrayList<String> src = new ArrayList<String>();
    ArrayList<String> trg = new ArrayList<String>();
    ArrayList<Integer> k = new ArrayList<Integer>();

    // 1. test series
    src.add(" Tthisisastring    abc");
    k.add(-1);
    trg.add("");
    
    src.add(" Tthisisastring    abc");
    k.add(0);
    trg.add("");
    
    src.add(" Tthisisastring    abc");
    k.add(1);
    // still empty as the string starts with a space
    trg.add("");
    
    src.add(" Tthisisastring    abc");
    k.add(2);
    trg.add(" Tthisisastring");
    
    src.add(" Tthisisastring\tabc");
    k.add(2);
    trg.add(" Tthisisastring");
    
    src.add(" Tthisisastring\n\rabc");
    k.add(2);
    trg.add(" Tthisisastring");
        
    src.add(" Tthisisastring    abc");
    k.add(3);
    trg.add(" Tthisisastring    abc");
    
    src.add(" Tthisisastring    abc");
    k.add(4);
    trg.add(" Tthisisastring    abc");
      
    // 2d. test series
    src.add("This is a stupid    string\tha-ha");
    k.add(-1);
    trg.add("");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(0);
    trg.add("");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(1);
    trg.add("This");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(2);
    trg.add("This is");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(3);
    trg.add("This is a");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(4);
    trg.add("This is a stupid");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(5);
    trg.add("This is a stupid    string");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(6);
    trg.add("This is a stupid    string\tha-ha");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(7);
    trg.add("This is a stupid    string\tha-ha");
    
    src.add("This is a stupid    string\tha-ha");
    k.add(100);
    trg.add("This is a stupid    string\tha-ha");
    
    // 3d. test series
    
    assertEquals(src.size(), trg.size());
    assertEquals(src.size(), k.size());
    
    for (int i = -1; i < 10; ++i) {
      src.add("");
      k.add(i);
      trg.add("");
    }
    
    for (int i = 0; i < src.size(); ++i) {
      String s = src.get(i);
      String tExp = trg.get(i);
      String tAct = StringUtils.truncAtKthWhiteSpaceSeq(s, k.get(i));
      
      if (tExp.compareTo(tAct) != 0) {
        System.out.println(String.format("Source '%s' k=%d Expected target: '%s' got: '%s'", s, k.get(i), tExp, tAct));
      }
      
      assertEquals(tExp, tAct);
    }
  }

}
