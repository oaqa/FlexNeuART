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

import static org.junit.Assert.assertTrue;

import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.junit.Test;

import edu.cmu.lti.oaqa.flexneuart.utils.ExtendedIndexEntry;
import edu.cmu.lti.oaqa.flexneuart.utils.JSONUtils;

public class JSONLUtilsTest {

  void oneDecodeTest(String textJSON,
               String[] keysStr, String[] valsStr,
               String[] keysStrArr, String[][] valsStrArr) {
    HashMap<String, String> str2strMap = new HashMap<String, String>();
    HashMap<String, ArrayList<String>> str2strArrMap = new HashMap<String, ArrayList<String>>();
    
    for (int i = 0; i < keysStr.length; ++i) {
      str2strMap.put(keysStr[i], valsStr[i]);
    }
    for (int i = 0; i < keysStrArr.length; ++i) {
      str2strArrMap.put(keysStrArr[i], new ArrayList<String>(Arrays.asList(valsStrArr[i])));
      
    }
    
    ExtendedIndexEntry e = null;
    try {
      e = JSONUtils.parseJSONIndexEntry(textJSON);
    } catch (Exception e2) {
      // TODO Auto-generated catch block
      e2.printStackTrace();
      assertTrue(false);
    }
    
    assertTrue(str2strMap.equals(e.mStringDict));
    assertTrue(str2strArrMap.equals(e.mStringArrDict));
    
  }
  
  @Test
  public void test1() {
    String doc = "{\n" + 
        "\"DOCNO\" : \"doc1\",\n" + 
        "\"text\" : \"val1\"\n" + 
        "}";
    String keys[] = {"DOCNO", "text"};
    String vals[] = {"doc1", "val1"};
    String keysArr[] = {};
    String valsArr[][] = {};
    oneDecodeTest(doc, keys, vals, keysArr, valsArr);
  }
  
  @Test
  public void test2() {
    String doc = "{\n" + 
        "\"DOCNO\" : \"doc2\",\n" + 
        "\"text\" : \"val2\",\n" + 
        "\"answer_list\" : [\"1\",\"2\",\"3\"],\n" +
        "\"answer_list2\" : [\"11\",\"22\",\"33\"]\n" +
        "}";
    String keys[] = {"DOCNO", "text"};
    String vals[] = {"doc2", "val2"};
    String keysArr[] = {"answer_list", "answer_list2"};
    String valsArr[][] = {{"1", "2", "3"}, {"11", "22", "33"}};
    oneDecodeTest(doc, keys, vals, keysArr, valsArr);
  }

}
