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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A data entry wrapper class, which is basically a dictionary
 * where keys are strings and values are strings, arrays of strings,
 * or binary arrays. Each entry has an optional ID.
 * 
 * @author Leonid Boytsov
 *
 */
public class DataEntryFields {
  public DataEntryFields(String entryId) {
    mEntryId = entryId;
  }
  public final String                    mEntryId;
  
  /**
   * Check if there's a non-null value for the given field name.
   * 
   * @param fieldName  a field name
   * @return true, if the key exists and the value is not null.
   */
  boolean hasField(String fieldName) {
    return mObjDict.get(fieldName) != null;
  }
  
  // Set functions assign field values
  void setString(String fieldName, String value) {
    mObjDict.put(fieldName, value);
  }
  void setBinary(String fieldName, byte [] value) {
    mObjDict.put(fieldName, value);
  }
  void setStringArray(String fieldName, String[] value) {
    mObjDict.put(fieldName, value);
  }
  
  // Get functions retrieve values and do runtime type checking
  public String getString(String fieldName) {
    Object value = mObjDict.get(fieldName);
    if (value == null) return null;
    if (!(value instanceof String)) {
      throw new RuntimeException("Field '" + fieldName + "' is not string!");
    }
    return (String) value;
  }
  
  public String getStringDefault(String fieldName, String defaultValue) {
    String value = getString(fieldName);
    if (value == null) {
      return defaultValue;
    }
    return value;
  }
  
  public byte[] getBinary(String fieldName) {
    Object value = mObjDict.get(fieldName);
    if (value == null) return null;
    if (!(value instanceof byte[])) {
      throw new RuntimeException("Field '" + fieldName + "' is not binary array!");
    }
    return (byte[]) value;
  }
  
  public String[] getStringArray(String fieldName) {
    Object value = mObjDict.get(fieldName);
    if (value == null) return null;
    if (!(value instanceof String[])) {
      throw new RuntimeException("Field '" + fieldName + "' is not string array!");
    }
    return (String[]) value;
  }
  
  public void addAll(DataEntryFields othEntry) {
    for (Entry<String, Object> e : othEntry.mObjDict.entrySet()) {
      String key = e.getKey();
      if (hasField(key)) {
        throw new RuntimeException("addAll: repeating key: '" + key + "'");
      }
      mObjDict.put(key, e.getValue());
    }
  }
  
  private Map<String, Object>            mObjDict = new HashMap<String, Object>();
}
