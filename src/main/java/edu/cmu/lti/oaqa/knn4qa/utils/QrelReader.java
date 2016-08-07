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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.File;
import java.io.IOException;
import java.util.*;

import org.apache.commons.io.FileUtils;

public class QrelReader {
  private HashMap<String, HashMap<String, String>>    mQrels = 
                                      new HashMap<String, HashMap<String, String>>(); 
  
  /*
   *  Reads TREC qrels from a file.
   */
  public QrelReader(String fileName) throws IOException {
    for (String s: FileUtils.readLines(new File(fileName))) {
      s = s.trim();
      if (s.isEmpty()) continue;
      String parts[] = s.split("\\s+");
      String queryId = parts[0];
      String docId   = parts[2];
      String rel     = parts[3];
      HashMap<String, String>   val0 = mQrels.get(queryId);
      if (val0 == null) {
        val0 = new HashMap<String, String>();
        mQrels.put(queryId, val0);
      }
      val0.put(docId, rel);
    }
  }
  
  /**
   * Retrieves relevance value.
   * 
   * @param queryId query identifier
   * @param docId   document identifier
   * @return qrel value for given parameters, or null, if not respective value
   * exists.
   */
  public String get(String queryId, String docId) {
    HashMap<String, String>   val0 = mQrels.get(queryId);
    if (null == val0) return null;
    return val0.get(docId);
  }
  /**
   * Return all relevance values for a given query.
   * 
   * @param queryId query identifier
   * @return all qrels in the form of a map: docId -> qrel value
   */
  public HashMap<String, String> getQueryQrels(String queryId) {
    return mQrels.get(queryId);
  }
}
