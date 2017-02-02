/*
 *  Copyright 2017 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.annotators;

import com.google.gson.*;

import java.io.*;


public class JSONAnnotData {
  public String            id;
  public JSONAnnotation[]  annotations;
  
  public static void main(String argv[]) throws Exception {
    Gson gson = new Gson();
    
    BufferedReader r = new BufferedReader(new FileReader(new File(argv[0])));
    String s;
    
    while ((s = r.readLine()) != null) {    
      JSONAnnotData  j = gson.fromJson(s, JSONAnnotData.class);
      
      System.out.println(j.id);
      System.out.println("========================");
      for (JSONAnnotation a : j.annotations) {
        System.out.println(a.type + " " + a.label + " [" + a.start + " , " + a.end + ")");
      }
      System.out.println("========================");
    }

  }
}
