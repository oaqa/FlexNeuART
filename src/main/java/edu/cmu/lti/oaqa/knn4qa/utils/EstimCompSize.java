/*
 *  Copyright 2015 Carnegie Mellon University
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

import java.io.*;
import java.util.zip.*;

import org.apache.tools.bzip2.CBZip2InputStream;

/**
 *   Provides rough estimates for compressed/uncompressed files.
 */
public class EstimCompSize {
  
  /**
   * A very rough estimate of the compression ratio for an XML file.
   * 
   * @param fileName    input file
   * @return            compression estimate (&lt; 1 smaller means better compression).
   */
  public static float estimCompressRatio(String fileName) {
    if (fileName.endsWith(".gz")) return 0.25f;
    if (fileName.endsWith(".bz2")) return 0.2f;
    return 1.0f;
  }
  
  /**
   * A very rough estimate of the size of a potentially compressed XML file.
   * @param fileName        input file
   * @return                an estimate of the uncompressed size.
   * @throws                IOException
   */
  public static long estimUncompSize(String fileName) throws IOException {
    File  f = new File(fileName);
    if (!f.exists() || !f.isFile()) {
      new IOException("File not found: '" + fileName + "'");
    }
    return (long)Math.ceil(f.length() / estimCompressRatio(fileName)); 
  }  
}
