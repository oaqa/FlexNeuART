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

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.google.gson.*;

/**
 * A bunch of useful functions to work with JSONL files that
 * keep data for indexing and querying. Each line in such JSONL file is 
 * a dictionary with string keys. Values are either strings,
 * or arrays of strings. Binary data is supported as well,
 * but it should be provided in a separate file.
 * 
 * @author Leonid Boytsov
 * 
 */
public class JSONUtils {
  
  /**
   * Read the next JSON entry from a JSONL file. 
   * 
   * @param inpText input text
   * @return next entry, or null, if no further entry can be found.
   * @throws IOException
   */
  public static String readNextJSONEntry(BufferedReader inpText, int recNo) throws IOException {
    
    while (true) {
      String docLine = inpText.readLine();
  
      if (docLine == null) return null;

      docLine = docLine.trim();
    
      // SKip empty lines
      if (docLine.isEmpty()) {
        continue;
      }
    
      return docLine;
    }
  }

  /**
   * Parses a string JSON entry like this one: 
   * {"DOCNO":"doc0","text":"This is a short text. This is a second text sentence."}
   * It is synchronized, because it uses a static variable: JSON parser.
   * 
   * @param doc     an input JSON entry
   * @param recNo   record number
   * 
   * @return a DataEntry object, which may have a null entryId.
   */
  public static synchronized DataEntryFields parseJSONEntry(String doc, int recNo) throws Exception {
    JsonElement root = mParser.parse(doc);
    JsonObject obj = root.getAsJsonObject();
    
    String entryId = null;
    
    for (Entry<String, JsonElement> e : obj.entrySet()) {
      String key = e.getKey();
      JsonElement val = e.getValue();
      if (key.compareTo(Const.DOC_ID_FIELD_NAME) == 0) {
        entryId = val.getAsString();
      }
    }

    DataEntryFields res = new DataEntryFields(entryId);

    for (Entry<String, JsonElement> e : obj.entrySet()) {
      String fieldName = e.getKey();
      JsonElement val = e.getValue();
      if (val.isJsonPrimitive()) {
        res.setString(fieldName, val.getAsString());
      } else if (val.isJsonArray()) {
        JsonArray arr = val.getAsJsonArray();
        ArrayList<String> tmpLst = new ArrayList<String>();
        for (JsonElement e1 : arr) {
          tmpLst.add(e1.getAsString());
        }
        res.setStringArray(fieldName, tmpLst.toArray(new String[tmpLst.size()]));
      } else {
        throw new Exception("Unsupported JSON entry # " + recNo + 
                            " invalid value for key: '" + fieldName + "' " +
                            "Invalid JSON: " + doc);
      }
    }
    return res;
  }
  
  /**
   * Generates a JSON entry that can be consumed by indexing/querying applications.
   * It is synchronized, because it uses a static variable: JSON parser.
   * 
   * @param fields  (key, value) pairs; key is a field name, value is a text of the field.
   * @return    a simple dictionary JSON entry.
   */
  public static synchronized String genJSONIndexEntry(Map <String,String> fields) {
    return mGson.toJson(fields, fields.getClass());
  }
  
  public static void main(String [] args) { 
    HashMap<String, String> e1 = new HashMap<String, String>();
    e1.put("DOCNO", "doc0");
    e1.put("text", "This is a short text. This is a second text sentence.");
    
    System.out.print(genJSONIndexEntry(e1));
  }
  
  final static Gson mGson = new Gson();
  final static JsonParser mParser = new JsonParser();
}
