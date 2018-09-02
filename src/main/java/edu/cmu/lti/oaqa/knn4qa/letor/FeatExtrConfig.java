/*
 *  Copyright 2018 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.util.Map;
import java.util.Map.Entry;

import com.github.andrewoma.dexx.collection.HashMap;
import com.google.gson.Gson;
import edu.cmu.lti.oaqa.annographix.util.MiscHelper;

class OneFeatExtrConf {
  String                  type;
  Map<String, String>     params;
  
  String getRequiredParam(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for the extractor %s",
          name, type));
    return val;
  }
  String getParam(String name, String defaultValue) {
    String val = params.get(name);
    return val != null ? val : defaultValue;
  }
  float getParam(String name, float defaultValue) {
    String val = params.get(name);
    return val != null ? Float.parseFloat(val) : defaultValue;
  } 
  int getParam(String name, int defaultValue) {
    String val = params.get(name);
    return val != null ? Integer.parseInt(val) : defaultValue;
  }  
}

public class FeatExtrConfig {

  public static String FIELD_NAME = "fieldName";
  public static String EXTR_TYPE = "extrType";
  
  public static String SIMIL_TYPE = "similType";
  
  OneFeatExtrConf[]   extractors;
  
  public static FeatExtrConfig readConfig(String inputFile) throws Exception {
    Gson gson = new Gson();    

    return gson.fromJson(MiscHelper.readFile(inputFile), 
                         FeatExtrConfig.class);
    
  }
  
  /**
   * Just a stupid testing function.
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    String inputFile = args[0];
   
    FeatExtrConfig tmp = readConfig(inputFile);
    
    for (OneFeatExtrConf extr:tmp.extractors) {
      System.out.println(extr.type);
      System.out.println(extr.params);
      for(Entry<String, String> e : extr.params.entrySet())
        System.out.println(e.getKey() + " -> " + e.getValue());
    }
  }  
}


