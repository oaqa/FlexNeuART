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

import org.junit.Test;

import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.JSONUtils;

public class JSONLUtilsTest {

  void oneDecodeTest(String textJSON,
               String   entryId,
               String[] keysStr, String[] valsStr,
               String[] keysStrArr, String[][] valsStrArr) {
    DataEntryFields e = null;
    try {
      e = JSONUtils.parseJSONEntry(textJSON, 0);
    } catch (Exception e2) {
      // TODO Auto-generated catch block
      e2.printStackTrace();
      assertTrue(false);
    }
    
    assertTrue(entryId.compareTo(e.mEntryId) == 0);
    
    for (int i = 0; i < keysStr.length; ++i) {
      String fieldName = keysStr[i];
      assertTrue(e.hasField(fieldName));
      assertTrue(e.getString(fieldName).compareTo(valsStr[i]) == 0);
    }
    for (int i = 0; i < keysStrArr.length; ++i) {
      String fieldName = keysStrArr[i];
      String expVal[] = valsStrArr[i];
      assertTrue(e.hasField(fieldName));
      String actualVal[] = e.getStringArray(fieldName);
      
      assertTrue(actualVal.length == expVal.length);
      for (int k = 0; k < expVal.length; ++k) {
        assertTrue(expVal[k].compareTo(actualVal[k]) == 0);
      }
    }
  
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
    oneDecodeTest(doc, "doc1", keys, vals, keysArr, valsArr);
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
    oneDecodeTest(doc, "doc2", keys, vals, keysArr, valsArr);
  }

}
