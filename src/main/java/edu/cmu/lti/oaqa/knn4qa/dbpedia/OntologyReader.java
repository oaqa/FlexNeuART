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

import com.hp.hpl.jena.rdf.model.*;
import com.hp.hpl.jena.vocabulary.RDFS;

import java.util.HashMap;

public class OntologyReader {

  public OntologyReader(String fileName) {
    final Model dbpedia = ModelFactory.createDefaultModel();
    dbpedia.read( "fileName", "RDF/XML" );
    StmtIterator stmts = dbpedia.listStatements(null, RDFS.subClassOf, (RDFNode) null);
    
    while ( stmts.hasNext() ) {
      final Statement stmt = stmts.next();
      System.out.println( stmt.getSubject() + " is a subclass of " + stmt.getObject() );
    }
  }
  
  public static void main(String[] args) {
    OntologyReader r = new OntologyReader(args[0]);
  }

}
