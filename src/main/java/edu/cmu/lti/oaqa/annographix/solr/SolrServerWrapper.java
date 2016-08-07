/*
 *  Copyright 2014 Carnegie Mellon University
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

package edu.cmu.lti.oaqa.annographix.solr;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

import org.apache.solr.client.solrj.*;
import org.apache.solr.client.solrj.SolrRequest.METHOD;
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.client.solrj.request.DirectXmlRequest;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.*;

/**
 * This is a considerably re-worked and expanded version of SolrWrapper from:
 * https://github.com/oaqa/solr-provider/ .
 *
 * <p>
 * It is different in several ways, most importantly:
 * </p>
 * <ol> 
 *   <li> 
 *      It has a function {@link #submitXML(String)}, 
 *      which allows one to pass arbitrary XML.
 *      This is useful for batch mode.
 *   </li>   
 *   <li>
 *     It doesn't support the embedded SOLR server (which seems to be a broken
 *     feature anyway).
 *   </li>
 *   <li>
 *     It doesn't apply an ad hoc escapeQuery.
 *   </li>
 *   <li>
 *      Some of the utility functions were placed into a separate class {@link SolrUtils}.
 *   </li>
 * </ol>
 * 
 * @author Leonid Boytsov   based largely on SolrWrapper code written
 *                          by Elmer Garduno and Alkesh Patel. 
 * 
 */
public final class SolrServerWrapper implements Closeable {
  /**
   * Opens a connection to a SOLR server.
   * 
   * @param serverUrl a full URI that includes a <b>core</b>. Unlike the SOLR
   *                  Web interface address, it doesn't contain '#'. 
   *                  For example, instead of http://localhost:8984/solr/#/AQUAINT
   *                  use: http://localhost:8984/solr/AQUAINT.
   * @throws Exception
   */
  public SolrServerWrapper(String serverUrl) throws Exception{
    this.mServer=createSolrServer(serverUrl);
  }

  /** 
   * Opens SOLR connection, creates an object 
   * of the type {@link org.apache.solr.client.solrj.SolrServer}. 
   */
  private SolrServer createSolrServer(String url) throws Exception {
    SolrServer server = new HttpSolrServer(url);
    // server.ping();
    return server;
  }

  /** 
   * @return the current SOLR-server object  
   *         of the type {@link org.apache.solr.client.solrj.SolrServer}. 
   * @throws Exception
   */

  public SolrServer getServer() throws Exception {
    return mServer;
  }

  /**
   * Executes a string query.
   * 
   * @param q        a query string.
   * @param numRet   the maximum number of entries to return.  
   * @return a list of documents, an object of the type {@link org.apache.solr.common.SolrDocumentList}. 
   * @throws SolrServerException
   */
  public SolrDocumentList runQuery(String q, int numRet)
      throws SolrServerException {
    SolrQuery query = new SolrQuery();
    query.setQuery(q);
    query.setRows(numRet);
    query.setFields("*", "score");
    QueryResponse rsp = mServer.query(query, METHOD.POST);
    return rsp.getResults();
  }

  /**
   * Executes a query, where the query is represented by an object
   * of the type {@link org.apache.solr.client.solrj.SolrQuery}.
   * 
   * @param query    a query object of the type {@link org.apache.solr.client.solrj.SolrQuery}.
   *   
   * @return a list of documents, an object of the type {@link org.apache.solr.common.SolrDocumentList}. 
   * @throws SolrServerException
   */
  public SolrDocumentList runQuery(SolrQuery query)
      throws SolrServerException {
    QueryResponse rsp = mServer.query(query);
    return rsp.getResults();
  }

  /**
   * Executes a query, additionally allows to specify result fields.
   * 
   * @param q           a query string.
   * @param fieldList   a list of field names.
   * @param results     the maximum number of entries to return.
   * 
   * @return a list of documents, which is an object of the type {@link org.apache.solr.common.SolrDocumentList}.
   * @throws SolrServerException
   */
  public SolrDocumentList runQuery(String q, 
                                  List<String> fieldList,
                                  int results) throws SolrServerException {
    SolrQuery query = new SolrQuery();
    query.setQuery(q);
    query.setRows(results);
    query.setFields(fieldList.toArray(new String[1]));
    QueryResponse rsp = mServer.query(query, METHOD.POST);
    return rsp.getResults();
  }
  
  /**
   * Executes a query, additionally allows to specify the default field, the filter query, AND result fields.
   * 
   * @param q               a query string.
   * @param defaultField    a default field name (or null).
   * @param fieldList       a list of field names.
   * @param filterQuery     a name of the filter query that can be applied without changing scores (or null).
   * @param results         the maximum number of entries to return.
   * 
   * @return a list of documents, which is an object of the type {@link org.apache.solr.common.SolrDocumentList}.
   * @throws SolrServerException
   */
  public SolrDocumentList runQuery(String q, 
                                  String       defaultField,
                                  List<String> fieldList,
                                  String       filterQuery,
                                  int results) throws SolrServerException {
    SolrQuery query = new SolrQuery();
    query.setQuery(q);
    if (filterQuery != null) query.setParam("fq", filterQuery);
    if (defaultField != null) query.setParam("df", defaultField);
    query.setRows(results);
    query.setFields(fieldList.toArray(new String[1]));
    QueryResponse rsp = mServer.query(query, METHOD.POST);
    return rsp.getResults();
  }  

  /**
   * Reads a value of a single- or multi-value text field from a SOLR server. 
   * 
   * @param docId         a document ID.
   * @param idField       a name of the ID field.
   * @param textFieldName a name of the text field whose value we need to obtain.
   * @return an array of field values, if the field is single-value, the array
   *         contains only one entry.
   * @throws SolrServerException
   */
  public ArrayList<String> getFieldText(String docId, 
                                        String idField, 
                                        String textFieldName) 
      throws SolrServerException {
    String q = idField + ":" + docId;
    SolrQuery query = new SolrQuery();
    query.setQuery(q);
    query.setFields(textFieldName);
    QueryResponse rsp = mServer.query(query);

    ArrayList<String> docText = null;
    if (rsp.getResults().getNumFound() > 0) {
      Object o = rsp.getResults().get(0).getFieldValues(textFieldName);
      if (o instanceof String) {
        docText = new ArrayList<String>();
        docText.add((String)o);
      } else {
        @SuppressWarnings({ "unchecked"})
        ArrayList<String> results = (ArrayList<String>)o; 
        docText = results;
      }
    }
    return docText;
  }
  /**
   * Submits a batch of update instructions in XML format.
   * 
   * @param    docXML input XML that represents one or more update instruction.
   * @throws Exception
   */ 
  public void submitXML(String docXML) throws Exception {
    DirectXmlRequest xmlreq = new DirectXmlRequest("/update", docXML);
    mServer.request(xmlreq);
  }

  /**
   * Converts an object of the type {@link org.apache.solr.common.SolrInputDocument} 
   * into XML to post over HTTP for indexing.
   * 
   * @param solrDoc a document to be indexed, which is represented by 
   *                an object of the type {@link org.apache.solr.common.SolrInputDocument}.
   * 
   * @return a textual representation of the document in XML format.
   * @throws Exception
   */
  public String convertSolrDocInXML(SolrInputDocument solrDoc)throws Exception{
    return ClientUtils.toXML(solrDoc);
  }

  /**
   * Index one document (in XML format), see {@link #convertSolrDocInXML(SolrInputDocument)}.
   * 
   * @param docXML a textual representation of the document in XML format.
   * @throws Exception
   */
  public void indexDocument(String docXML) throws Exception {

    String xml = "<add>" + docXML + "</add>";
    DirectXmlRequest xmlreq = new DirectXmlRequest("/update", xml);
    mServer.request(xmlreq);
  }

  /**
   * Index one document using a key-value map representation.
   * 
   * @param    keyValueMap  key-value map (field name, field text).
   * @throws Exception
   */
  public void indexDocument(HashMap<String,Object> keyValueMap) throws Exception {
    SolrInputDocument solrDoc = SolrUtils.buildSolrDocument(keyValueMap);
    String docXML=this.convertSolrDocInXML(solrDoc);
    String xml = "<add>" + docXML + "</add>";
    DirectXmlRequest xmlreq = new DirectXmlRequest("/update", xml);
    mServer.request(xmlreq);
  }

  /**
   * Issue a commit.
   * @throws Exception
   */
  public void indexCommit() throws Exception {
    mServer.commit();
  }

 /**
  * Delete documents that satisfy a given query.
  * 
  * @param query delete documents that satisfy this query.
  * @throws Exception
  */
  public void deleteDocumentByQuery(String query)throws Exception{

    mServer.deleteByQuery(query);
  }

  @Override
  public void close() throws IOException {
    // Nothing to do here, just a stub
  }
  
  /** 
   * A {@link org.apache.solr.client.solrj.SolrServer} variable that we
   * wrap.
   */
  private final SolrServer mServer;
}
