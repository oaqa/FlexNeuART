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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map.Entry;

import org.bson.BsonBinaryReader;
import org.bson.BsonReader;
import org.bson.BsonType;

/**
 * A bunch of useful functions to work with binary files that
 * keep binary data for indexing and querying. Such file
 * contain multiple BSON entries prefixed by the entry length. 
 * Each BSON record is a dictionary with string keys. 
 * Values are binary arrays except for the document ID, which is
 * supposed to be a string.
 * 
 * @author Leonid Boytsov
 * 
 */
public class BSONUtils {
  /**
   * Read the next BSON entry from the input stream. Each entry is supposed to be
   * preceded 
   * 
   * @param mInpBin  input data stream
   * @param recNo    the number of the current record (for error reporting)
   * 
   * @return a byte array containing an unparsed BSON data or null if we reached the end of file.
   * @throws IOException
   */
  static byte [] readNextBSONEntry(DataInputStream mInpBin, int recNo) throws IOException {
    byte entrySizeBuffData[] = new byte[4];
    int readQty = mInpBin.read(entrySizeBuffData);
    if (readQty == -1) {
      return null;
    }
    if (readQty != 4) {
      throw new RuntimeException("Truncated binary file entry #: " + recNo + 
                                " expected 4 bytes, got " + readQty);
    }
    ByteBuffer entrySizeBuff = ByteBuffer.wrap(entrySizeBuffData);
    entrySizeBuff.order(Const.BYTE_ORDER);
    int binQty = entrySizeBuff.getInt();
    byte binEntryData[] = new byte[binQty];
    readQty = mInpBin.read(binEntryData);
    if (readQty != binQty) {
      throw new RuntimeException("Truncated binary file entry #: " + recNo + recNo +
                                " expected " + binQty + " bytes, got " + readQty);
    }
    return binEntryData;
  }
  
  /**
   * Parse a key-value dictionary encoded as a BSON entry.
   * Keys are strings and values are binary except for an entry ID,
   * which is a string.
   * 
   * @param entryData
   * @param recNo
   * @return
   * @throws Exception
   */
  public static DataEntryFields parseBSONEntry(byte [] entryData, int recNo) throws Exception {
    try (BsonReader breader = new BsonBinaryReader(ByteBuffer.wrap(entryData))) {
      String entryId = null;
      HashMap<String, byte[]>  tmpRes = new HashMap<String, byte[]>();
      String fieldName;
      
      breader.readStartDocument();   
      while (breader.readBsonType() != BsonType.END_OF_DOCUMENT) {
        fieldName = breader.readName();
        if (fieldName.compareTo(Const.DOC_ID_FIELD_NAME) == 0) {
          if (breader.getCurrentBsonType() != BsonType.STRING) {
            throw new RuntimeException("The type of the field '" + fieldName + "' should be string!");
          }
          entryId = breader.readString();
        } else {
          if (breader.getCurrentBsonType() != BsonType.BINARY) {
            throw new RuntimeException("The type of the field '" + fieldName + "' should be binary!");
          }         
          tmpRes.put(fieldName, breader.readBinaryData().getData());
          break;
        }
      }
      breader.readEndDocument(); 
      
      DataEntryFields res = new DataEntryFields(entryId);

      for (Entry<String, byte[]> e : tmpRes.entrySet()) {
        res.setBinary(e.getKey(), e.getValue());
      }
      
      return res;
    }
  }
  
  public static void main(String [] args) throws Exception {
   
    DataInputStream inp = new DataInputStream(new FileInputStream(args[0]));
    DataEntryFields dataEntry = parseBSONEntry(readNextBSONEntry(inp, 1), 1);

    System.out.println(dataEntry.mEntryId);
    float [] fvec = BinReadWriteUtils.readPackedDenseVector(dataEntry.getBinary(args[1]));
    for (int i = 0; i < fvec.length; i++)
      System.out.println(fvec[i] );

//    byte curr[];
//    int line = 0;
//    while ((curr = readNextBSONEntry(inp, ++line)) != null) {
//      if (line % 10000 == 0) {
//        System.out.println(line);
//      }
//      DataEntry dataEntry = parseBSONEntry(curr, line);
//    }
//    System.out.println(line);
  }
}
