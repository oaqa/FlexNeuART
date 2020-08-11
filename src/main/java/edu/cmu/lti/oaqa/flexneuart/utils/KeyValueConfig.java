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
      throw new Exception("The parametr dictionary is missing!");
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
    return Float.parseFloat(val);
  } 
  
  public int getReqParamInt(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    return Integer.parseInt(val);
  }
  
  public boolean getReqParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null)
      throw new Exception(String.format("Mandatory parameter %s is undefined for " + getName(), name));
    return Boolean.parseBoolean(val);
  } 
  
  public boolean getParamBool(String name) throws Exception {
    String val = params.get(name);
    if (val == null) return false;
    return Boolean.parseBoolean(val);
  }
  
  public String getParam(String name, String defaultValue) {
    String val = params.get(name);
    return val != null ? val : defaultValue;
  }
  
  public float getParam(String name, float defaultValue) {
    String val = params.get(name);
    return val != null ? Float.parseFloat(val) : defaultValue;
  } 
  
  public int getParam(String name, int defaultValue) {
    String val = params.get(name);
    return val != null ? Integer.parseInt(val) : defaultValue;
  }
  
  public boolean getParam(String name, boolean defaultValue) {
    String val = params.get(name);
    return val != null ? Boolean.parseBoolean(val) : defaultValue;
  }

}