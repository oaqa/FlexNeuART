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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;

/**
 * An auto-closable resource class that writes possibly compressed entries in either JSONL or 
 * series-of-XML entries format. If, after removing the .gz or bz2. suffix,
 * the file has a .txt extension, it is assumed to be a series of XML entries.
 * If we move an optional .gz or .bz2 suffix and obtain a .jsonl suffix,
 * the input is assumed to be in a JSONL format.
 * 
 * @author Leonid Boytsov
 *
 */
public class DataEntryWriter implements java.lang.AutoCloseable {

  /**
   * A constructor that guesses the data type from the file name.
   * 
   * @param fileName  input file
   * @throws IllegalArgumentException
   * @throws IOException
   */
  public DataEntryWriter(String fileName) throws IllegalArgumentException, IOException {
    mIsJson = DataEntryReader.isFormatJSONL(fileName);
    mOut = new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(fileName))); 
  }
  
  public void writeEntry(Map<String, String> fieldInfo) throws IOException, ParserConfigurationException, TransformerException {
    String doc = null;
    if (mIsJson) {
      doc = JSONUtils.genJSONIndexEntry(fieldInfo);
    } else {
      doc = mXmlHlp.genXMLIndexEntry(fieldInfo);
    }
    mOut.write(doc);
    mOut.write(NL);
  }
  

  @Override
  public void close() throws Exception {
    mOut.close();
  }
    
  final private boolean mIsJson;
  final private BufferedWriter mOut;
  final private XmlHelper mXmlHlp = new XmlHelper();
  
  final static private String NL = System.getProperty("line.separator");

}
