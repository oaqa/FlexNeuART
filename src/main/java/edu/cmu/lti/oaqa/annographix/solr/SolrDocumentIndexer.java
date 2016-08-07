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

import java.io.StringWriter;
import java.util.Map;
import java.util.Map.Entry;

import org.w3c.dom.Document;
import org.w3c.dom.DocumentFragment;
import org.w3c.dom.Element;

import edu.cmu.lti.oaqa.annographix.solr.UtilConst;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

/**
 * 
 * This is a helper class that reads annotated documents and
 * submit indexing requests to a SOLR server.
 * 
 * @author Leonid Boytsov
 * 
 */
public class SolrDocumentIndexer implements DocumentIndexer {
  private static final boolean DEBUG_PRINT = false;

  /**
   * 
   * @param solrURI     an address of the server that includes the core, e.g., http://localhost:8984/solr/AQUAINT
   * @param textField   a name of the annotated text field.
   * @param annotField  a name of the annotation field to store annotations for textField.
   * @throws Exception
   */
  public SolrDocumentIndexer(String solrURI, 
                             String textField, 
                             String annotField) throws Exception {
    mTextFieldName = textField;
    mAnnotFieldName = annotField;
    
    mDocFactory           = DocumentBuilderFactory.newInstance();
    mDocFactory.setValidating(false);
    mDocBuilder           = mDocFactory.newDocumentBuilder();
    mTransformerFactory   = TransformerFactory.newInstance();
    mTransformer          = mTransformerFactory.newTransformer();
    mTargetServer         = new SolrServerWrapper(solrURI);
  }


  @Override
  /**
   * The function indexes provided SOLR documents. It assumes that
   * annotations have the same document no/id and are sorted.
   * 
   * @param docFields   (key, value) pairs; key is a field name, value is a text of the field.
   * @param annots      the list of annotations for the annotated field.
   * 
   * @throws Exception 
   * 
   */
  public void consumeDocument(Map<String, String> docFields,
                              OffsetAnnotationFileEntry[]  annots) 
                              throws Exception{  
    /*
     * Check if the annotation entries' array is valid:
     * 1) Annotations should be sorted.
     * 2) She should all represent the same document number.
     */
 
    for  (int i = 1; i< annots.length; ++i) {
      OffsetAnnotationFileEntry prevAnnot = annots[i-1],
                      currAnnot = annots[i];
      if (!prevAnnot.mDocNo.equals(currAnnot.mDocNo)) {
        throw new Exception(String.format("Bug: docnos different, i=%d, prev='%s', curr='%s'",
                                          i, prevAnnot.mDocNo, currAnnot.mDocNo));
      }
      if (prevAnnot.compareTo(currAnnot) > 0) {
        throw new Exception(String.format("Bug: sortedness violation, i=%d, docNo='%s'",
                                          i, prevAnnot.mDocNo));
      }
    }
    
    String origDocText = docFields.get(mTextFieldName);
    String docNo       = docFields.get(UtilConst.TAG_DOCNO);
    
    final String docText = UtilConst.
                  replaceWhiteSpaces(UtilConst.removeBadUnicode(origDocText)).    
                  replace("?", " ");
    /*
     * The previous transformation replaces some characters with spaces, 
     * but it shouldn't change the string length or positions or regular characters!
     * Let's make a simple sanity check proving that this is true:
     */
    if (docText.length() != origDocText.length())
      throw new Exception("Bug: Character replacement procedure changed the length of the string!");
    
    /*
     * Don't change the size of the string: 
     * This will screw annotation offsets.
     */
    if (docText.length() != origDocText.length()) {
      throw new Exception("Bug: the size of the document was changed during a transformation!");
    }
    
    
    if (null == mBatchXML) {
      mBatchXML  = mDocBuilder.newDocument();
      mAddNode   = mBatchXML.createElement("add");
      mBatchXML.appendChild(mAddNode);
    }
    
    DocumentFragment FragRoot = mBatchXML.createDocumentFragment();
    Element oneDoc = mBatchXML.createElement("doc");
    FragRoot.appendChild(oneDoc);
    
    addField(oneDoc, UtilConst.ID_FIELD, docNo);
    /*
     * It will be the SOLR responsibility to stem the words
     * properly. Here, we will only stem keywords stored with
     * annotations.
     */
    addField(oneDoc, mTextFieldName, docText);
    
    for (Entry<String, String> e: docFields.entrySet()) {
      String key = e.getKey(), value = e.getValue();
      if (!key.equalsIgnoreCase(UtilConst.ID_FIELD) &&
          !key.equalsIgnoreCase(UtilConst.TAG_DOCNO) &&
          !key.equalsIgnoreCase(mTextFieldName)) {
        addField(oneDoc, key, value);
      }
    }

   
    // Create annotation representation
    StringBuilder annotString = new StringBuilder();
    
    for (OffsetAnnotationFileEntry e: annots) {
      // Replace potential occurrences of the payload char
      String annotLabel = UtilConst.removeBadUnicode(e.mLabel);
      
      if (annotLabel.indexOf(UtilConst.PAYLOAD_CHAR) >= 0) {
        throw new Exception("docNo = " + docNo + " annotation labels shouldn't " +
                            " annotation id: " + e.mAnnotId +       
                            " contain symbols '" + UtilConst.PAYLOAD_CHAR + "'" +
                            " offending annotation label: '" + e.mLabel + "'");
      }
        
      createPayloadStr(annotString,
          e,
          e.mStartChar,
          e.mStartChar + e.mCharLen,
          /*
           *  Let's enforce lowercasing of annotation labels,
           *  we do the same in a query plugin. 
           */
          annotLabel.toLowerCase()
      );                
    }
    
    addField(oneDoc, mAnnotFieldName, annotString.toString()); 
    
    mAddNode.appendChild(FragRoot);    
  }
  
   /**
   * 
   * This function  creates a payload string in the format that can
   * be understood by the indexing application.
   * 
   * @param annotString   A buffer where to we append the result.
   * @param e             An annotation entry read from an Indri-style annotation file.
   * @param itStart       A start position. 
   * @param itEnd         An end position.
   * @param annotLabel    An annotation label 
   * 
   *  @throws Exception
   */
  private void createPayloadStr(
                             StringBuilder   annotString,
                             OffsetAnnotationFileEntry e,
                             int      iStart,
                             int      iEnd,
                             String   annotLabel
                             ) throws Exception {
    StringBuilder oneAnnot = new StringBuilder();    
    
    
    oneAnnot.append(annotLabel);
    oneAnnot.append(UtilConst.PAYLOAD_CHAR);
    
    oneAnnot.append(iStart);
    oneAnnot.append(UtilConst.PAYLOAD_ID_SEP_CHAR);
    oneAnnot.append(iEnd);
    oneAnnot.append(UtilConst.PAYLOAD_ID_SEP_CHAR);      
    oneAnnot.append(e.mAnnotId);
    oneAnnot.append(UtilConst.PAYLOAD_ID_SEP_CHAR);
    oneAnnot.append(e.mParentId);
    oneAnnot.append(' ');
    
    String str = oneAnnot.toString();
                
    /*
     * Annotation label shouldn't be too long! 
     */
    int maxLen = UtilConst.MAX_WORD_LEN;
    if (str.length() > maxLen) {
      throw new Exception(String.format("The payload is longer than %d," + 
                                        "one has to use shorter annotation labels, payload = '%s'",
                                        maxLen, str));  
    }
    
    annotString.append(str);
  }
  
  private void addField(Element oneDoc, String fieldName, String fieldValue) {
    Element  XmlDocId = mBatchXML.createElement("field");
    XmlDocId.setAttribute("name", fieldName);
    XmlDocId.setTextContent(fieldValue);
    oneDoc.appendChild(XmlDocId);
  }
  
  String mSolrURI;

  @Override
  public void sendBatch() throws Exception {
    if (mBatchXML == null) return;
    
    System.out.println("Sending batch!");
    
    StringWriter OutStr = new StringWriter();
    StreamResult result = new StreamResult(OutStr);
  
    DOMSource source = new DOMSource(mBatchXML);
    mTransformer.transform(source, result);

    String xml = OutStr.getBuffer().toString();

    System.out.println("Submitting XML");

    if (DEBUG_PRINT) System.out.println(xml);
    
    mTargetServer.submitXML(xml);
    System.out.println("Issuing a commit");
    mTargetServer.indexCommit();
    System.out.println("Batch is submitted!");    
    
    mBatchXML = null;
    mAddNode = null;
  }
  
  private String                  mTextFieldName;
  private String                  mAnnotFieldName;
  
  private DocumentBuilderFactory  mDocFactory;
  private DocumentBuilder         mDocBuilder;
  private TransformerFactory      mTransformerFactory;
  private Transformer             mTransformer;
  
  private SolrServerWrapper       mTargetServer;
  private Document                mBatchXML = null;
  private Element                 mAddNode = null;
}
