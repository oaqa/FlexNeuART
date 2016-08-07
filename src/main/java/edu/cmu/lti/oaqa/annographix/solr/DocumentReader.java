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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.xml.sax.SAXException;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;
import edu.cmu.lti.oaqa.annographix.util.XmlHelper;
import edu.cmu.lti.oaqa.annographix.solr.UtilConst;

/**
 * This class reads documents and their respective annotations
 * from files previously created by an annotation pipeline.
 * 
 * @author Leonid Boytsov
 *
 */
public class DocumentReader {
  /**
   * Reads documents and annotations from respective text files.
   * 
   * @param docTextFile   file with documents, 
   *                       one document in Indri format, 
   *                       inside &lt;DOC&gt;...&lt;/DOC&gt;.
   *                         
   * @param docAnnotFile  file with annotations in Indri format.
   * @param textFieldName a name of the text field.
   * @param batchQty      a batch size.
   * @param obj           a document consumer (e.g., it reads files and 
   *                      indexes them in SOLR).
   * @throws Exception 
   */
  public static void readDoc(
                 String docTextFile,
                 String textFieldName,
                 String docAnnotFile, 
                 int batchQty, 
                 DocumentIndexer obj) 
                     throws Exception {
    BufferedReader  inpText = new BufferedReader(
        new InputStreamReader(CompressUtils.createInputStream(docTextFile)));
    BufferedReader  inpAnnot = new BufferedReader(
        new InputStreamReader(CompressUtils.createInputStream(docAnnotFile)));
    
    OffsetAnnotationFileEntry prevEntry = null;
    
    String docText = XmlHelper.readNextXMLIndexEntry(inpText);
    
    XmlHelper xmlHlp = new XmlHelper();
    
    for (int docNum = 1; 
         docText!= null; 
         ++docNum, docText = XmlHelper.readNextXMLIndexEntry(inpText)) {
      
      // 1. Read document text
      Map<String, String> docFields = null;
            
      try {
        docFields = xmlHlp.parseXMLIndexEntry(docText);
      } catch (SAXException e) {
        System.err.println("Parsing error, offending DOC:" + NL + docText);
        throw new Exception("Parsing error.");
      }

      String docText4Anot = docFields.get(textFieldName); 
          
      if (docText4Anot == null) {
        System.err.println("Parsing error, offending DOC:" + NL + docText);
        throw new Exception("Can't find the field: '" + docText4Anot + "'");
      }
      
      String docno = docFields.get(UtilConst.TAG_DOCNO);
      
      if (docno == null) {
        System.err.println("Parsing error, offending DOC:" + NL + docText);
        throw new Exception("Can't find the field: '" + 
                            UtilConst.TAG_DOCNO + "'");
      }

      
      // 2. Read document annotations
      ArrayList<OffsetAnnotationFileEntry> annotList = new ArrayList<OffsetAnnotationFileEntry>();
      
      
      while (prevEntry == null || prevEntry.mDocNo.equals(docno)) {
        String annotLine = null;
        if (prevEntry == null) {
          annotLine = inpAnnot.readLine();
          
          if (null == annotLine) break;
          
          try {
            prevEntry = OffsetAnnotationFileEntry.parseLine(annotLine);
          } catch (NumberFormatException e) {
            throw new Exception("Failed to parse annotation line: '" 
                                  + annotLine + "', exception: " + e);
          } catch (EntryFormatException e) {
            throw new Exception("Failed to parse annotation line: '" 
                                  + annotLine + "' exception" + e );
          }                  
        }
        
        if (prevEntry.mDocNo.equals(docno)) {
          annotList.add(prevEntry);
        } 
        
        /*
         *  Don't clear prevEntry in this case:
         *  we will need it in the following documents.
         */
        if (!prevEntry.mDocNo.equals(docno)) {
          break;
        }
        /*
         *  However, we need to discard the annotation
         *  entry that represents an already processed
         *  or skipped document. 
         *  
         */
        prevEntry = null;
      }
      
      /*
       *  3. pass a parsed document + annotation to the indexer
       */
      
      OffsetAnnotationFileEntry[] annots = new OffsetAnnotationFileEntry[annotList.size()];
      for (int i = 0; i < annots.length; ++i)
        annots[i] = annotList.get(i);
      // we must short annotations
      Arrays.sort(annots);
      
      obj.consumeDocument(docFields, annots);
      
      if (docNum % batchQty == 0) obj.sendBatch();
    }
    
    obj.sendBatch();
    inpAnnot.close();
    inpText.close();
  }
  private final static String NL = System.getProperty("line.separator");
}
