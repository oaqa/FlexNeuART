/*
 *  Copyright 2017 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.dbpedia;

import org.apache.jena.rdf.model.*;
import org.apache.jena.vocabulary.RDFS;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.HashMap;
import java.io.*;

public class OntologyReader {
  private static final String ROOT_CLASS = "thing";
  
  private static final Logger logger = LoggerFactory.getLogger(OntologyReader.class);
  HashMap<String,HashSet<String>> mSuperClasses = new HashMap<String,HashSet<String>>();
  HashMap<String, String>         mParents = new HashMap<String, String>();
  HashMap<String, Boolean>        mIsSubClass = new HashMap<String, Boolean>();
  
  public OntologyReader(String fileName) throws FileNotFoundException {
    Model dbpedia = ModelFactory.createDefaultModel();
    dbpedia.read(new FileInputStream(new File(fileName)), "RDF/XML" );

    StmtIterator stmts = dbpedia.listStatements(null, RDFS.subClassOf, (RDFNode) null);

    int qty = 0;
    
    while ( stmts.hasNext() ) {
      final Statement stmt = stmts.next();
      String subClass = stmt.getSubject().getLocalName();
      String superClass = stmt.getObject().asResource().getLocalName();
      
      mParents.put(subClass, superClass);
      logger.info( subClass + " is a subclass of " + superClass);
      qty++;
    }
    logger.info("Read " + qty + " DBPedia concepts");
  }
  
  /**
   * Verifies if one is a subclass of another.
   * 
   * @param subClass
   * @param superClass
   * @return
   */
  public boolean isSubClass(String subClass, String superClass) {
    String key = combine(subClass, superClass);
    Boolean res = mIsSubClass.get(key);
    if (res != null) return res;
    
    String parent = mParents.get(subClass);

    if (superClass.equalsIgnoreCase(ROOT_CLASS)) res = true; else res = false;
    
    while (parent != null && !parent.equalsIgnoreCase(ROOT_CLASS)) {
      if (parent.equalsIgnoreCase(superClass)) {
        res = true;
        break;
      }
      parent = mParents.get(parent);
    }
    
    mIsSubClass.put(key, res); // memoization
    return res;
  }
  
  private String combine(String subClass, String superClass) {
    return subClass + "_" + superClass;
  }

  public static void main(String[] args) {
    OntologyReader r = null;
    try {
      r = new OntologyReader(args[0]);
    } catch (Exception e) {
      e.printStackTrace();
    }
    System.out.println("Positive examples:");
    System.out.println(r.isSubClass("Thing",       "Thing"));
    System.out.println(r.isSubClass("RugbyPlayer", "Thing"));
    System.out.println(r.isSubClass("RugbyPlayer", "Person"));
    System.out.println(r.isSubClass("RugbyPlayer", "Athlete"));
    System.out.println(r.isSubClass("Athlete",     "Person"));
    System.out.println(r.isSubClass("Biathlete",   "WinterSportPlayer"));
    
    
    System.out.println("Negative examples:");
    System.out.println(r.isSubClass("Thing", "RugbyPlayer"));
    System.out.println(r.isSubClass("WinterSportPlayer", "Biathlete"));
    System.out.println(r.isSubClass("City",        "Location")); // Surprisingly but this is the case in at least 2015 version
    System.out.println(r.isSubClass("Country",     "Location")); // Again, country isn't a location, weird!
  }

}
