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

import org.apache.commons.io.FileUtils;
import org.junit.Test;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.PersistentKeyValBackend;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.LuceneDbBackend;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.MapDbBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;


public class BackendTest {
  public final static String INDEX_PREFIX = "index_prefix";
  
  void testOneBackend(PersistentKeyValBackend backend, int qty) throws Exception {
    Path tmpDir = java.nio.file.Files.createTempDirectory(null);
    
    String indexPrefix = tmpDir + File.pathSeparator + INDEX_PREFIX;
    backend.initIndexForWriting(indexPrefix, qty);
    
    HashMap<String, String> data = new HashMap<String, String>();
    
    for (int i = 0; i < qty; i++) {
      String key = "key" + i;
      String value = "value" + i;
      
      backend.put(key, value.getBytes());
      data.put(key, value);
    }
    
    backend.close();
    
    backend.openIndexForReading(indexPrefix);
    
    String keys[] = backend.getKeyArray();
    
    assertEquals(keys.length, data.size());
    
    for (int i = 0; i < keys.length; i++) {
      byte[] binValue = backend.get(keys[i]);
      assertTrue(binValue != null);
      String value = new String(binValue);
      assertTrue(value.compareTo(data.get(keys[i])) == 0);
    }
    
    FileUtils.deleteDirectory(tmpDir.toFile());
  }
  
  @Test
  public void testLuceneBackend() throws Exception {
    for (int qty = 0; qty < 10 * 1000 ; qty += 2000) {
      testOneBackend(new LuceneDbBackend(), qty);
    }
  }
  
  @Test
  public void testMapDbBackend() throws Exception {
    for (int qty = 0; qty < 10 * 1000 ; qty += 2000) {
      testOneBackend(new MapDbBackend(), qty);
    }
  }
}
