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

import java.io.*;
import java.util.Map;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateProvider;

public class GenGalagoQueries {

  public static void main(String[] args) {
    if (args.length != 2 && args.length != 3) {
      System.err.println("Specify args: <input> <output> <optional maximum number of queries>");
      System.exit(1);
    }
    
    try {    
      int maxNumQuery = Integer.MAX_VALUE;
      if (args.length == 3) maxNumQuery = Integer.parseInt(args[2]);

      BufferedWriter  outText = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(args[1])));
      
      outText.write("{\ncasefold : true,\nqueries :\n");
      
      BufferedReader  inpText = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(args[0])));

      Map<String, String>    queryData = null;
      
      String docText = XmlHelper.readNextXMLIndexEntry(inpText);        
      int queryNum = 0;
      for (; docText!= null && queryNum < maxNumQuery; 
          docText = XmlHelper.readNextXMLIndexEntry(inpText)) {

        // 1. Parse a query
        try {
          queryData = XmlHelper.parseXMLIndexEntry(docText);
        } catch (Exception e) {
          System.err.println("Parsing error, offending DOC:\n" + docText);
          throw new Exception("Parsing error.");
        }
        
        String queryID = queryData.get(CandidateProvider.ID_FIELD_NAME);
        if (null == queryID) {
          throw new Exception(
              String.format("Query id (%s) is undefined for query # %d",
                  CandidateProvider.ID_FIELD_NAME, queryNum));
        }        
        
        String text = queryData.get(CandidateProvider.TEXT_FIELD_NAME);
        if (null == text) {
          throw new Exception(
              String.format("Query (%s) is undefined for query # %d",
                  CandidateProvider.TEXT_FIELD_NAME, queryNum));
        }
        
        text = text.trim();
        
        if (queryNum > 0) outText.write(","); else outText.write("\n");
        outText.write(String.format("{\n\"number\" :\n \"%s\",\n \"text\" : \"#combine(%s)\"\n}", queryID, text));

        queryNum++;

      }
      
      outText.write("\n}");
      outText.close();
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
  }

}
