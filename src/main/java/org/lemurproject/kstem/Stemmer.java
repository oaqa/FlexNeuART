/*
 *  BSD License (http://lemurproject.org/galago-license)
 */
package org.lemurproject.kstem;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
/**
 * 
 * 
 * @author sjh
 */
public abstract class Stemmer {

  // each instance of Stemmer should have it's own lock
  final Object lock = new Object();
  long cacheLimit = 50000;
  HashMap<String, String> cache = new HashMap();

  public String stem(String term) {
    if (cache.containsKey(term)) {
      return cache.get(term);
    }
    String stemmedTerm;
    
    synchronized (lock) {
      stemmedTerm = stemTerm(term);
    }
    
    if (!cache.containsKey(stemmedTerm)) {
      cache.put(term, stemmedTerm);
    }
    if (cache.size() > cacheLimit) {
      cache.clear();
    }
    return stemmedTerm;
  }
  // This function should only be use synchronously (see 'lock')
  protected abstract String stemTerm(String term);
}
