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
import java.io.*;

public class OntologyReader {
  private static final Logger logger = LoggerFactory.getLogger(OntologyReader.class);
  HashSet<String> mIsSubClassOf = new HashSet<String>();
  
  public OntologyReader(String fileName) throws FileNotFoundException {
    Model dbpedia = ModelFactory.createDefaultModel();
    dbpedia.read(new FileInputStream(new File(fileName)), "RDF/XML" );
    StmtIterator stmts = dbpedia.listStatements(null, RDFS.subClassOf, (RDFNode) null);

    int qty = 0;
    
    while ( stmts.hasNext() ) {
      final Statement stmt = stmts.next();
      String subClass = stmt.getSubject().getLocalName();
      String superClass = stmt.getObject().asResource().getLocalName();
      mIsSubClassOf.add(combine(subClass, superClass));
      //System.out.println( subClass + " is a subclass of " + superClass + " " + isSubClass(subClass, superClass));
      qty++;
    }
    logger.info("Read " + qty + " DBPedia concepts");
  }
  
  public boolean isSubClass(String subClass, String superClass) {
    return mIsSubClassOf.contains(combine(subClass, superClass));
  }
  
  private String combine(String subClass, String superClass) {
    return subClass + "_" + superClass;
  }

  public static void main(String[] args) {
    try {
      OntologyReader r = new OntologyReader(args[0]);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}
