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
 * A helper class to parse JSON configuration files. This parser is slightly restricted 
 * in that it does not permit mixing primitive and nested structures in the same array.
 * It is also "lazy" in that it defers some processing to the moment the user requests
 * specific data elements. Perhaps, a better approach would be doing some schema validation
 * but it is not done right now.
 * 
 * @author Leonid Boytsov
 *
 */
public class RestrictedJsonConfig {
  /**
   * A constructor that accepts a reference to an already parsed JSON element.
   * 
   * @param root
   * @throws Exception
   */
  protected RestrictedJsonConfig(JsonElement root, String configName) throws Exception {
    mConfigName = configName;
    
    if (root.isJsonObject()) {
      JsonObject obj = root.getAsJsonObject();
      mDataDict = new HashMap<String, JsonElement>();
      for (Entry<String, JsonElement> e : obj.entrySet()) {
        mDataDict.put(e.getKey(), e.getValue());
      }
    } else if (root.isJsonArray()) {
      mDataArr = new ArrayList<JsonElement>();
      JsonArray arr = root.getAsJsonArray();
      for (int i = 0; i < arr.size(); i++) {
        mDataArr.add(arr.get(i));
      }
    } else {
      throw new Exception("Bug: a config element '" + configName + "' is neither an array nor a key-value collection!");
    }
  }
  
  /**
   * The function reads the configuration file. 
   * 
   * @param configName   configuration type/name
   * @param configFileNmae     configuration file name
   * @return   an instance of the type RestrictedJsonConfig
   * 
   * @throws Exception 
   */
  public static RestrictedJsonConfig readConfig(String configName, String configFileName) throws Exception {
    JsonParser parser = new JsonParser();

    String configText = FileUtils.readFileToString(new File(configFileName), Const.ENCODING);
    return new RestrictedJsonConfig(parser.parse(configText), configName);
  }
  
  public String getName() {
    return mConfigName;
  }
  
  
  public boolean isArray() {
    return mDataArr != null;
  }
  
  public boolean isKeyValCollection() {
    return mDataDict != null;
  }
  
  private RestrictedJsonConfig getParamNestedConfig(JsonElement val, String name) throws Exception {
    String elemName = mConfigName + "/" + name;
    if (!val.isJsonObject() && !val.isJsonArray()) {
      throw new Exception("Element '" + elemName + "' is not a nested config!");
    }
    return new RestrictedJsonConfig(val, elemName);
  }
  
  public RestrictedJsonConfig getReqParamNestedConfig(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      throw new Exception(String.format("Mandatory nested configuration parameter %s is undefined for " + getName(), name));
    return getParamNestedConfig(val, name);
  }
  
  public RestrictedJsonConfig getParamNestedConfig(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      return null;
    return getParamNestedConfig(val, name);
  }
 
  public String getReqParamStr(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      throw new Exception(String.format("Mandatory string parameter '%s' is undefined for " + getName(), name));
    return getParamStr(val, name);
  }
  
  public float getReqParamFloat(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      throw new Exception(String.format("Mandatory float parameter '%s' is undefined for " + getName(), name));
    return getParamFloat(val, name);
  } 
  
  public int getReqParamInt(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      throw new Exception(String.format("Mandatory integer parameter '%s' is undefined for " + getName(), name));
    return getParamInt(val, name);
  }
  
  public boolean getReqParamBool(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null)
      throw new Exception(String.format("Mandatory boolean parameter '%s' is undefined for " + getName(), name));
    return getParamBool(val, name);
  }
  
  private JsonElement getValueElemByKey(String name) throws Exception {
    if (mDataDict == null) {
      throw new Exception("Requested key: '" + name + 
                          "' but the config '" + mConfigName + "' does not have a key-value collection");       
    }
    return mDataDict.get(name);
  }
  
  public boolean getParamBool(String name) throws Exception {
    JsonElement val = getValueElemByKey(name);
    if (val == null) return false;
    return getParamBool(val, name);
  }
  
  public String getParam(String name, String defaultValue) throws Exception {
    JsonElement val = getValueElemByKey(name);
    return val != null ? getParamStr(val, name) : defaultValue;
  }
  
  public float getParam(String name, float defaultValue) throws Exception {
    JsonElement val = getValueElemByKey(name);
    return val != null ? getParamFloat(val, name) : defaultValue;
  } 
  
  public int getParam(String name, int defaultValue) throws Exception {
    JsonElement val = getValueElemByKey(name);
    return val != null ? getParamInt(val, name) : defaultValue;
  }
  
  public boolean getParam(String name, boolean defaultValue) throws Exception {
    JsonElement val = getValueElemByKey(name);
    return val != null ? getParamBool(val, name) : defaultValue;
  }
  
  private void checkIfArray() throws Exception {
    if (mDataArr == null) {
      throw new Exception("Requested array of primitives, but the config '" + mConfigName + "' does not have an array!");       
    }    
  }
  
  public RestrictedJsonConfig[] getParamConfigArray() throws Exception {
    checkIfArray();
    
    int qty = mDataArr.size();
    RestrictedJsonConfig res[] = new RestrictedJsonConfig[qty];
    
    for (int i = 0; i < qty; i++) {
      JsonElement val = mDataArr.get(i);
      String elemName = mConfigName + "[" + i + "]";
      res[i] = new RestrictedJsonConfig(val, elemName); 
    }
    return res;
  }
  
  public String[] getParamStringArray() throws Exception {
    checkIfArray();
    
    int qty = mDataArr.size();
    String res[] = new String[qty];
    
    for (int i = 0; i < qty; i++) {
      JsonElement val = mDataArr.get(i);
      String elemName = mConfigName + "[" + i + "]";
      checkIfPrimitive(elemName, val);
      try {
        res[i] = val.getAsString();
      } catch (ClassCastException e) {
        throw new Exception("Element '" + elemName + "' is not a string!");
      } 
    }
    return res;
  }
  
  public float [] getParamFloatArray() throws Exception {
    checkIfArray();
    
    int qty = mDataArr.size();
    float res[] = new float[qty];
    
    for (int i = 0; i < qty; i++) {
      JsonElement val = mDataArr.get(i);
      String elemName = mConfigName + "[" + i + "]";
      checkIfPrimitive(elemName, val);
      try {
        res[i] = val.getAsFloat();
      } catch (ClassCastException e) {
        throw new Exception("Element '" + elemName + "' is not a float!");
      } 
    }
    return res;
  }
  
  public int [] getParamIntArray() throws Exception {
    checkIfArray();
    
    int qty = mDataArr.size();
    int res[] = new int[qty];
    
    for (int i = 0; i < qty; i++) {
      JsonElement val = mDataArr.get(i);
      String elemName = mConfigName + "[" + i + "]";
      checkIfPrimitive(elemName, val);
      try {
        res[i] = val.getAsInt();
      } catch (ClassCastException e) {
        throw new Exception("Element '" + elemName + "' is not an int!");
      } 
    }
    return res;
  }
  
  public boolean [] getParamBoolArray() throws Exception {
    checkIfArray();
    
    int qty = mDataArr.size();
    boolean res[] = new boolean[qty];
    
    for (int i = 0; i < qty; i++) {
      JsonElement val = mDataArr.get(i);
      String elemName = mConfigName + "[" + i + "]";
      checkIfPrimitive(elemName, val);
      try {
        res[i] = val.getAsBoolean();
      } catch (ClassCastException e) {
        throw new Exception("Element '" + elemName + "' is not a boolean!");
      } 
    }
    return res;
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

  
  private HashMap<String, JsonElement>    mDataDict = null; // Data dictionary elements can be either primitives 
  private ArrayList<JsonElement>          mDataArr = null;
  
  private final String                    mConfigName;
}
