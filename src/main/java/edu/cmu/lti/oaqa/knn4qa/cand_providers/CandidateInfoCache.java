/*
 *  Copyright 2016 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.*;
import java.io.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CandidateInfoCache {
  private static final String INCOMPLETE_SUFFIX = ".incomplete";
  
  private static final Logger logger = LoggerFactory.getLogger(CandidateInfoCache.class);
 
  private static String getIncompleteFlagFileName(String cacheName) {
    return cacheName + INCOMPLETE_SUFFIX;
  }
  
  /**
   *  Checks if the case exists and the incomplete-flag file is deleted.
   */
  public static boolean cacheExists(String cacheName) {
    String incompleteName = getIncompleteFlagFileName(cacheName);
    
    return  (new File(cacheName)).exists() && 
            !(new File(incompleteName)).exists();
  }
  
  public void writeCache(String cacheName) throws Exception {
    synchronized(this) {    
      String incompleteName = getIncompleteFlagFileName(cacheName);
      
      FileOutputStream tagOut = new FileOutputStream(incompleteName);
      tagOut.close();
      
      FileOutputStream fileOut = new FileOutputStream(cacheName);
      ObjectOutputStream out = new ObjectOutputStream(fileOut);
    
      int qty = mResultCache.size();
      out.writeObject(qty);
      for (Map.Entry<String, CandidateInfo> e : mResultCache.entrySet()) {
        out.writeObject(e.getKey());
        out.writeObject(e.getValue());
      }
      
      out.close();
      if (!(new File(incompleteName)).delete()) {
        throw new Exception("Cannot delete file: " + incompleteName);
      }
      logger.info(String.format("Wrote %d entries to file %s",  qty, cacheName));
    }
  }
  
  public void readCache(String cacheName) throws IOException, ClassNotFoundException {
    synchronized(this) {    
      mResultCache.clear();
      FileInputStream fileIn = new FileInputStream(cacheName);    
      ObjectInputStream in = new ObjectInputStream(fileIn);
      
      Integer qty = (Integer)in.readObject();
      for (int i = 0; i < qty; ++i) {
        String          queryId = (String)in.readObject();
        CandidateInfo   info = (CandidateInfo)in.readObject();
        addOrReplaceCacheEntry(queryId, info);
      }
      
      in.close();
      logger.info(String.format("Read %d entries from file %s",  qty, cacheName));
    }
  }
  
  public void addOrReplaceCacheEntry(String queryId, CandidateInfo info) {
    synchronized(this) {
      mResultCache.put(queryId, info);
    }
  };
  
  public CandidateInfo getCacheEntry(String queryID) {
    synchronized(this) {
      return mResultCache.get(queryID);
    }
  }
  
  private TreeMap<String,CandidateInfo> mResultCache = new TreeMap<String,CandidateInfo>();

}
