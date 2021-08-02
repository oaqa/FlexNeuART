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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.util.Iterator;

/**
 * 
 * An abstract class to hide implementation details of one or more persistent
 * key-value store backends. The key-value store operates in two modes: writing (append-only),
 * and reading.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class PersistentKeyValBackend {
  // Open the previously create index
  public abstract void openIndexForReading(String indexPrefix) throws Exception;
  /**
   * / Initialize the index for writing
   * 
   * @param indexPrefix  an index file(s) prefix. For some backends it can be a complete file name.
   * @param expectedQty  an expected number of entries. This parameter is not used by all backends.
   * @throws Exception
   */
  public abstract void initIndexForWriting(String indexPrefix, int expectedQty) throws Exception;
  
  /**
   * Close the index and save it to disk if necessary.
   * 
   * @throws Exception
   */
  public abstract void close() throws Exception;
  
  public abstract void put(String key, byte [] value) throws Exception;
  public abstract byte[] get(String key) throws Exception;
  
  /**
   * @return an array of keys
   */
  public abstract String[] getKeyArray() throws Exception;
  
  public abstract Iterator<String> getKeyIterator() throws Exception;
  
  /**
   * @return a number of stored elements
   */
  public abstract int size() throws Exception;
}
