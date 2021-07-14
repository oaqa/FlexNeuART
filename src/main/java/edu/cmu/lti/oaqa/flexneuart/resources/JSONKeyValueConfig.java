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
package edu.cmu.lti.oaqa.flexneuart.resources;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.cmu.lti.oaqa.flexneuart.utils.Const;

/**
 * A helper class to process key-value configurations files. It supports extracting primitive JSON
 * values by the key name. For non-primitive values, a downstream code can extract the reference
 * to the underlying JSonElement and process it as needed. The latter should permit nested configuration
 * files.
 * <p>
 * Our key-value configuration files come in two flavors: 
 * <ul>
 *  <li>A single dictionary
 *  <li>An array of dictionaries (e.g., this is how we specify extractors)
 * </ul>
 * 
 * @author Leonid Boytsov
 *
 */
public class JSONKeyValueConfig {
  /**
   * A constructor that accepts a reference to an already parsed JSON element. We expect it
   * to be of the JsonObject type, i.e., the underlying JSON should be a dictionary.
   * 
   * @param root
   * @throws Exception
   */
  protected JSONKeyValueConfig(JsonElement root, String configName) throws Exception {
    if (!root.isJsonObject()) {
      throw new Exception("The JSON root element is not an object!");
    }
    JsonObject obj = root.getAsJsonObject();
    mConfigName = configName;
    
    for (Entry<String, JsonElement> e : obj.entrySet()) {
      mDataDict.put(e.getKey(), e.getValue());
    }
  }
  
  /**
   * The function reads the configuration files and determines if it has an array of key-value
   * configs or a single key-value config. It then returns a result. 
   * 
   * @param configName   configuration type/anme
   * @param configFileNmae     configuration file name
   * @return an array of configuration objects: if the JSON has only a single key-value dictionary-style
   *         config, the array has only one element.
   * @throws Exception 
   */
  public static ArrayList<JSONKeyValueConfig> readConfig(String configName, String configFileName) throws Exception {
    JsonParser parser = new JsonParser();
    ArrayList<JSONKeyValueConfig> res = new ArrayList<JSONKeyValueConfig>();
    String configText = FileUtils.readFileToString(new File(configFileName), Const.ENCODING);
    JsonElement parsed = parser.parse(configText);
    
    if (parsed.isJsonArray()) {
      JsonArray arr = parsed.getAsJsonArray();
      for (int i = 0; i < arr.size(); i++) {
        res.add(new JSONKeyValueConfig(arr.get(i), configName));
      }
    } else {
      res.add(new JSONKeyValueConfig(parsed, configName));
    }
    
    return res;
  }
  
  public String getName() {
    return mConfigName;
  }
  
 
  public String getReqParamStr(String name) throws Exception {
    JsonElement val = mDataDict.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory string parameter %s is undefined for " + getName(), name));
    return getParamStr(val, name);
  }
  
  public float getReqParamFloat(String name) throws Exception {
    JsonElement val = mDataDict.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory float parameter %s is undefined for " + getName(), name));
    return getParamFloat(val, name);
  } 
  
  public int getReqParamInt(String name) throws Exception {
    JsonElement val = mDataDict.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory integer parameter %s is undefined for " + getName(), name));
    return getParamInt(val, name);
  }
  
  public boolean getReqParamBool(String name) throws Exception {
    JsonElement val = mDataDict.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory boolean parameter %s is undefined for " + getName(), name));
    return getParamBool(val, name);
  } 
  
  public boolean getParamBool(String name) throws Exception {
    JsonElement val = mDataDict.get(name);
    if (val == null) return false;
    return getParamBool(val, name);
  }
  
  public String getParam(String name, String defaultValue) throws Exception {
    JsonElement val = mDataDict.get(name);
    return val != null ? getParamStr(val, name) : defaultValue;
  }
  
  public float getParam(String name, float defaultValue) throws Exception {
    JsonElement val = mDataDict.get(name);
    return val != null ? getParamFloat(val, name) : defaultValue;
  } 
  
  public int getParam(String name, int defaultValue) throws Exception {
    JsonElement val = mDataDict.get(name);
    return val != null ? getParamInt(val, name) : defaultValue;
  }
  
  public boolean getParam(String name, boolean defaultValue) throws Exception {
    JsonElement val = mDataDict.get(name);
    return val != null ? getParamBool(val, name) : defaultValue;
  }
  
  private void checkIfPrimitive(String name, JsonElement val) throws Exception {
    if (!val.isJsonPrimitive()) {
      throw new Exception("Parameter '" + name + "' is not a JSON primitive!");
    }
  }
  
  private String getParamStr(JsonElement val, String name) throws Exception {
    // If the check succeeds the only possible exception is the class cast exception
    checkIfPrimitive(name, val);
    try {
      return val.getAsString();
    } catch (ClassCastException e) {
      throw new Exception("Parameter '" + name + "' is not a string!");
    }   
  }
  
  private float getParamFloat(JsonElement val, String name) throws Exception {
    // If the check succeeds the only possible exception is the class cast exception
    checkIfPrimitive(name, val);
    try {
      return val.getAsFloat();
    } catch (ClassCastException e) {
      throw new Exception("Parameter '" + name + "' is not a float!");
    }    
  }
  
  private int getParamInt(JsonElement val, String name) throws Exception {
    // If the check succeeds the only possible exception is the class cast exception
    checkIfPrimitive(name, val);
    try {
      return val.getAsInt();
    } catch (ClassCastException e) {
      throw new Exception("Parameter '" + name + "' is not an integer!");
    }
  }
  
  private boolean getParamBool(JsonElement val, String name) throws Exception {
    // If the check succeeds the only possible exception is the class cast exception
    checkIfPrimitive(name, val);
    try {
      return val.getAsBoolean();
    } catch (ClassCastException e) {
      throw new Exception("Parameter '" + name + "' is not a boolean (true/false)!");
    }    
  }
 
  
  /**
   * 
   * @param key a string key.
   * @return a JSON element for a given key.
   */
  JsonElement getJsonElement(String key) {
    return mDataDict.get(key);
  }
  
  private HashMap<String, JsonElement>  mDataDict = new HashMap<String, JsonElement>();
  private final String                  mConfigName;
}
