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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.*;


/**
 * A bunch of useful functions to work with JSONL files to
 * store indexable data. 
 * 
 * @author Leonid Boytsov
 * 
 */
public class JSONUtils {

  /**
   * Parses a string JSON entry like this one: 
   * {"DOCNO":"doc0","text":"This is a short text. This is a second text sentence."}
   * 
   * @param doc     an input JSON entry
   * @return a map with (key, value) pairs.
   */
  public static HashMap<String, String> parseJSONIndexEntry(String doc) {
    return mGson.fromJson(doc, HashMap.class);
  }
  
  /**
   * Generates a JSON entry that can be consumed by indexing/querying applications.
   * 
   * @param fields  (key, value) pairs; key is a field name, value is a text of the field.
   * @return    a simple dictionary JSON entry.
   */
  public static String genJSONIndexEntry(Map <String,String> fields) {
    return mGson.toJson(fields, fields.getClass());
  }
  
  /**
   * Read the next JSON entry from a JSONL file.
   * 
   * @param inpText input text
   * @return next entry, or null, if no further entry can be found.
   * @throws IOException
   */
  public static String readNextJSONEntry(BufferedReader inpText) throws IOException {
    String docLine = inpText.readLine();

    if (docLine == null || docLine.isEmpty()) return null;

    return docLine.trim();
  }
  
  public static void main(String [] args) {
    String doc1 = "{\n" + 
        "\"DOCNO\" : \"doc1\",\n" + 
        "\"text\" : \"val1\"\n" + 
        "}";
    String doc2 = "{\n" + 
        "\"DOCNO\" : \"doc2\",\n" + 
        "\"text\" : \"val2\"\n" + 
        "}";
    Map<String, String> e = parseJSONIndexEntry(doc2);
    System.out.println(e.get("DOCNO"));
    System.out.println(e.get("text"));
    
    e = new HashMap<String, String>();
    e.put("DOCNO", "doc0");
    e.put("text", "This is a short text. This is a second text sentence.");
    
    System.out.print(genJSONIndexEntry(e));
    System.out.println("###");
  }
  
  final static Gson mGson = new Gson();
}
