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

import java.io.File;
import java.util.Map;
import java.util.Map.Entry;

import com.google.gson.Gson;

import edu.cmu.lti.oaqa.knn4qa.utils.Const;

import org.apache.commons.io.FileUtils;

class OneFeatExtrConf {
  String                  type;
  Map<String, String>     params;
  
  String getReqParamStr(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for the extractor %s",
          name, type));
    return val;
  }
  float getReqParamFloat(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for the extractor %s",
          name, type));
    return Float.parseFloat(val);
  } 
  int getReqParamInt(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for the extractor %s",
          name, type));
    return Integer.parseInt(val);
  }
  
  boolean getReqParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for the extractor %s",
          name, type));
    return Boolean.parseBoolean(val);
  } 
  
  boolean getParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null) return false;
    return Boolean.parseBoolean(val);
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
  boolean getParam(String name, boolean defaultValue) {
    String val = params.get(name);
    return val != null ? Boolean.parseBoolean(val) : defaultValue;
  }
}

public class FeatExtrConfig {

  
  OneFeatExtrConf[]   extractors;
  
  public static FeatExtrConfig readConfig(String inputFile) throws Exception {
    Gson gson = new Gson();    

    return gson.fromJson(FileUtils.readFileToString(new File(inputFile), Const.ENCODING), 
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


