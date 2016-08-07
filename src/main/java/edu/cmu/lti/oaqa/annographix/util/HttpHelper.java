/*
 *  Copyright 2014 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.annographix.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;

public class HttpHelper {
  /**
   * Retrieves a page with a given URI.
   * 
   * @param uri a page URI
   * @return a page textual content.
   * 
   * @throws Exception
   */
  public static String get(String uri) throws Exception {
    URL obj = new URL(uri);
    HttpURLConnection con = (HttpURLConnection) obj.openConnection();
    
    // optional default is GET
    con.setRequestMethod("GET");
 
    //add request header
    con.setRequestProperty("User-Agent", UtilConst.USER_AGENT);
    
    int responseCode = con.getResponseCode();
    if (responseCode != 200) {
      throw new Exception("Cannot read using URI: " + uri);
    }
    
    BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
    String         respLine;
    StringBuffer   resp = new StringBuffer();
 
    while ((respLine = in.readLine()) != null) {
      resp.append(respLine);
      resp.append(NL);
    }
    
    in.close();
    
    return resp.toString();    
  }  
  private final static String NL = System.getProperty("line.separator");
}
