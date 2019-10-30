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
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.io.*;
import java.util.zip.*;

import org.apache.tools.bzip2.CBZip2InputStream;
import org.apache.tools.bzip2.CBZip2OutputStream;

import org.apache.commons.io.IOUtils;

/**
 *   Creates an input/output stream for a potentially compressed file;
 *   Determines a compression format by file extension.
 *   <p>
 *   Supports the following formats:
 *   </p>
 *   <ul>
 *   <li>For reading: .gz and bz2
 *   <li>For writing: only .gz
 *   </ul>
 *
 */
public class CompressUtils {
  
  /**
   * Creates an input stream to read from a regular or compressed file.
   * 
   * @param fileName a file name with an extension (.gz or .bz2) or without it;
   *                   if the user specifies an extension .gz or .bz2,
   *                   we assume that the input
   *                   file is compressed.
   * @return an input stream to read from the file. 
   * @throws IOException
   */
  public static InputStream createInputStream(String fileName) throws IOException {
    InputStream finp = new FileInputStream(fileName);
    if (fileName.endsWith(".gz")) return new GZIPInputStream(finp);
    if (fileName.endsWith(".bz2")) {
      finp.read(new byte[2]); // skip the mark

      return new CBZip2InputStream(finp);
    }
    return finp;
  }
  
  /**
   * Creates an output stream to write to a regular or compressed file.
   * 
   * @param fileName    a file name with an extension .gz or without it;
   *                    if the user specifies an extension .gz, we assume
   *                    that the output file should be compressed.
   * @return an output stream to write to a file.
   * @throws IOException
   */
  public static OutputStream createOutputStream(String fileName) throws IOException {
    OutputStream foutp = new FileOutputStream(fileName);
    if (fileName.endsWith(".gz")) return new GZIPOutputStream(foutp);
    if (fileName.endsWith(".bz2")) {
      throw new IOException("bz2 is not supported for writing");      
    }
    return foutp;
  }
  
  /**
   * A wrapper function that tries to compress the string using GZIP. However,
   * if the compressed string is longer than the original one,
   * we keep the string uncompressed. This resolves the issue
   * with short strings where compressed string is actually longer
   * than an uncompressed one.
   * 
   * @param input   input stream
   * @return  a byte array that represents the compressed string
   * @throws IOException
   */
  public static byte[] comprStr(String input) throws IOException {
  	ByteArrayOutputStream res = new ByteArrayOutputStream();
  	byte[] gzipped = gzipStr(input);
  	byte[] orig = input.getBytes(Const.ENCODING);
  	byte[] marker = {1};
  	if (gzipped.length < orig.length) {
  		marker[0] = 1;
  		res.write(marker);
  		res.write(gzipped);
  	} else {
  		marker[0] = 0;
  		res.write(marker);
  		res.write(orig);
  	}
  	return res.toByteArray();
  }

  /** 
   * Decompress a string previously compressed by {@link comprStr}
   * 
   * @param input a byte buffer with the compressed output
   * @return uncompressed string
   * @throws IOException
   */
  public static String decomprStr(byte [] input) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    if (input.length < 1) {
    	throw new IOException("Input is too small!");
    }
    byte marker = input[0];
    if (marker == 1) {
	    try{
	        IOUtils.copy(new GZIPInputStream(new ByteArrayInputStream(input, 1, input.length - 1)), out);
	    } catch(IOException e){
	        throw new RuntimeException(e);
	    }
	   
	    return new String(out.toByteArray(), Const.ENCODING);
    } else {
    	return new String(input, 1, input.length - 1, Const.ENCODING);
    }
  }
  
  /**
   * Compress the input string using the GZIP algorithm. String is
   * written in the {@link Const.ENCODING} encoding. Based on the
   * Largely based on https://piotrga.wordpress.com/2009/06/08/howto-compress-or-decompress-byte-array-in-java/
   * 
   * @param input   input stream
   * @return  a byte array that represents the compressed string
   * @throws IOException 
   */
  public static byte[] gzipStr(String input) throws IOException {
  	 ByteArrayOutputStream arrayOutputStream = new ByteArrayOutputStream();
     GZIPOutputStream gzipOutputStream = new GZIPOutputStream(arrayOutputStream);
     gzipOutputStream.write(input.getBytes(Const.ENCODING));
     gzipOutputStream.close();
     return arrayOutputStream.toByteArray();
  }
  
  /**
   * Compress the input string using BZIP2 algorithm. String is
   * written in the {@link Const.ENCODING} encoding. Based on the
   * Largely based on https://piotrga.wordpress.com/2009/06/08/howto-compress-or-decompress-byte-array-in-java/
   * 
   * @param input   input stream
   * @return  a byte array that represents the compressed string
   * @throws IOException 
   */
  public static byte[] bzip2Str(String input) throws IOException {
  	 ByteArrayOutputStream arrayOutputStream = new ByteArrayOutputStream();
  	 CBZip2OutputStream bzip2OutputStream = new CBZip2OutputStream(arrayOutputStream);
     bzip2OutputStream.write(input.getBytes(Const.ENCODING));
     bzip2OutputStream.close();
     return arrayOutputStream.toByteArray();
  }
  
  /**
   * Decompress string compressed by GZIP assuming the string encoding is {@link Const.ENCODING}.
   * Largely based on https://piotrga.wordpress.com/2009/06/08/howto-compress-or-decompress-byte-array-in-java/
   * 
   * @param input
   * @return
   * @throws IOException
   */
  public static String ungzipStr(byte[] input) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try{
        IOUtils.copy(new GZIPInputStream(new ByteArrayInputStream(input)), out);
    } catch(IOException e){
        throw new RuntimeException(e);
    }
    return new String(out.toByteArray(), Const.ENCODING);
  }
  
  /**
   * Decompress string compressed by BZIP2 assuming the string encoding is {@link Const.ENCODING}.
   * Largely based on https://piotrga.wordpress.com/2009/06/08/howto-compress-or-decompress-byte-array-in-java/
   * 
   * @param input
   * @return
   * @throws IOException
   */
  public static String unbzip2Str(byte[] input) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try{
        IOUtils.copy(new CBZip2InputStream(new ByteArrayInputStream(input)), out);
    } catch(IOException e){
        throw new RuntimeException(e);
    }
    return new String(out.toByteArray(), Const.ENCODING);
  }
  
  private static final String SUFF_ARRAY[] = {"", ".gz", ".bz2"};

  /**
   * Finds is there is a file in one of three forms: uncompressed, gz-compressed, bz2-compressed.
   * 
   * @param fileNamePrefix    file name prefix
   * @return
   * @throws Exception
   */
  public static String findFileVariant(String fileNamePrefix) throws Exception {
    String res = null;
    int cnt = 0;
    for (String suff : SUFF_ARRAY) {
      String fname = fileNamePrefix + suff;
      if ((new File(fname)).exists()) {
        if (0 == cnt++) res = fname;
      }
    }
    
    if (cnt == 0)
      throw new Exception("No file starting with " + fileNamePrefix + " is found");
    if (cnt > 1)
      throw new Exception("Multiple files starting with " + fileNamePrefix + " are found, e.g., compressed and uncompresseds");
    
    return res;
  }
  
  /** 
   * Removes a file suffix that is known to be created by a compression
   * program.
   * 
   * @param fileName  a file name
   * @return a string possibly without a suffix, e.g., fileName.txt if the input was fileName.txt.gz
   * 
   */
  public static String removeComprSuffix(String fileName) {
    
    for (String suff : SUFF_ARRAY) {
      if (!suff.isEmpty() && fileName.endsWith(suff)) {
        return fileName.substring(0, fileName.length() - suff.length());
      }
    }
    return fileName;
  }

} 