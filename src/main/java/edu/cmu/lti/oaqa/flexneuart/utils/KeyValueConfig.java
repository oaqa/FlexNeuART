package edu.cmu.lti.oaqa.flexneuart.utils;

import java.util.Map;

public abstract class KeyValueConfig {
  public abstract String getName();
	
  protected Map<String, String>     params;
  
  public Map<String, String> getAllParams() {
  	return params;
  }
  
  public String getReqParamStr(String name) throws Exception {
    if (params == null) {
      throw new Exception("The parameter dictionary is missing!");
    }
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    return val;
  }
  
  public float getReqParamFloat(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    try {
      return Float.parseFloat(val);
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not a float!");
    }
  } 
  
  public int getReqParamInt(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    try {
      return Integer.parseInt(val);
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not an int!");
    }
  }
  
  public boolean getReqParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    try {
      return Boolean.parseBoolean(val);
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not a boolean (true/false)!");
    }
  } 
  
  public boolean getParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null) return false;
    try {
      return Boolean.parseBoolean(val);
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not a boolean (true/false)!");
    }
  }
  
  public String getParam(String name, String defaultValue) {
    String val = params.get(name);
    return val != null ? val : defaultValue;
  }
  
  public float getParam(String name, float defaultValue) throws Exception {
    String val = params.get(name);
    try {
      return val != null ? Float.parseFloat(val) : defaultValue;
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not a float!");
    }
  } 
  
  public int getParam(String name, int defaultValue) throws Exception {
    String val = params.get(name);
    try {
      return val != null ? Integer.parseInt(val) : defaultValue;
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not an int!");
    }
  }
  
  public boolean getParam(String name, boolean defaultValue) throws Exception {
    String val = params.get(name);
    try {
      return val != null ? Boolean.parseBoolean(val) : defaultValue;
    } catch (NumberFormatException e) {
      throw new Exception("Parameter '" + name + "' is not a boolean (true/false)!");
    }
  }

}