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



/**
 * 
 * A simple helper class to read annotation descriptions from
 * an Indri-style offset annotation file.
 * 
 * @author Leonid Boytsov
 *
 */
public class OffsetAnnotationFileEntry 
                    implements Comparable<OffsetAnnotationFileEntry> {
  /*
   * Final immutable objects are constant!
   * They are assigned in the constructor and cannot
   * be changed afterwards.
   */  
  /** document number. */
  final String            mDocNo;
  /** annotation ID. */
  final int               mAnnotId;
  /** parent annotation ID. */
  final int               mParentId;
  /** optional annotation label. */
  final String            mLabel;
  /** optional annotation text. */
  final String            mText;
  /** the offset of the first annotation character. */
  final int               mStartChar;
  /** the length of annotation (in characters). */
  final int               mCharLen;  

  
  /**
    * Construct an entry from an Indri-style annotation offset file. 
    * 
    * @param line   a string of tab-separated field values.
    * @return an instance of the annotation class, which contains
    *         data extracted from tab-separated fields.
    */
  public static OffsetAnnotationFileEntry parseLine(String line) 
                                            throws 
                                            NumberFormatException,
                                            EntryFormatException {
    String parts[] = line.split("\\t");
    
    if (parts.length < 8) throw new EntryFormatException();
    
    return new OffsetAnnotationFileEntry(parts[0],
                                Integer.parseInt(parts[2]),
                                Integer.parseInt(parts[7]),
                                parts[3],
                                Integer.parseInt(parts[4]),
                                Integer.parseInt(parts[5]));
  }
    
  /**
   * Constructor for full-fledged annotations without corresponding text.
   * It should not be used directly: the object is created by the function
   * {@link #parseLine(String)}.
   * 
   * @param docNo       document number.
   * @param annotId     this annotation id.
   * @param parentId    parent annotation id.
   * @param label       annotation label.
   * @param startChar   the zero-based offset of the first annotation character.
   * @param charLen     the length of the annotation in characters.
   */
  private OffsetAnnotationFileEntry(String docNo, 
                         int annotId, 
                         int parentId, 
                         String label, 
                         int startChar, 
                         int charLen) {
    mDocNo      = docNo;
    mAnnotId    = annotId;
    mParentId   = parentId;
    mLabel      = label;
    mStartChar  = startChar;
    mCharLen    = charLen;
    mText       = null;
  }
  
  
  @Override
  public int compareTo(OffsetAnnotationFileEntry o) {
    if (!mDocNo.equals(o.mDocNo)) {
      return mDocNo.compareTo(o.mDocNo);
    }
    if (mStartChar != o.mStartChar) {
      return mStartChar - o.mStartChar;
    }
    /*
     *  If start offsets are equal, but the current object 
     *  has smaller size, it will go earlier
     */
    return mCharLen - o.mCharLen;
  }
}
