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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.io.*;


/**
 * A dumb class to efficiently process arbitrary XML files, where relatively small pieces 
 * of data enclosed by a pair of unique opening/closing XML tags follow one another.
 * 
 * <p>Parsing such an XML file using a SAX parser is quite a bit of pain in the neck.
 * In particular, frameworks such as UIMA support an iterative retrieval of 
 * data pieces, which is super-hard to do in the callback framework of a SAX
 * parser.</p>
 * 
 * <p>This class aims to carry out a shallow, but correct, parsing of the XML file. As
 * a result it extract verbatim text inside the specified tags <b>without an
 * attempt to deep-parse the contents inside this text.</b> The latter is
 * assumed to be done by, e.g., a standard DOM parser. However, we aim to
 * preserve all the tag attributes (including the enclosing tags) as well
 * as CDATA blocks.</p> 
 * 
 * @author Leonid Boytsov
 *
 */
public class XmlIterator {
  private static final char OPEN_BRACKET_CHAR = '<';
  private static final char CLOSE_BRACKET_CHAR = '>';
  private static final char SLASH_CHAR = '/';
  private static final String CDATA_TAG     = "![CDATA[";
  private static final String DOCTYPE_TAG   = "!DOCTYPE";
  private static final String PREAMBLE_TAG  = "?xml";

  
  /**
   * Constructor.
   * 
   * @param inpStream           an input string.
   * @param enclTagName         a non-empty enclosing tag (must be used only as an enclosing tag in the document).
   * @throws                    Exception
   */
  public XmlIterator(InputStream inpStream, String enclTagName) throws Exception {
    if (enclTagName.isEmpty()) 
      throw new Exception("Enclosing tag shouldn't be empty!");
    mEnclTagName = enclTagName;
    mInput = new BufferedReader(new InputStreamReader(inpStream));
  }
  
  enum TagType {
    TAG_REG_OPEN,
    TAG_REG_CLOSE,
    TAG_CDATA,
    TAG_DOCTYPE,
    TAG_PREAMBLE
  }
  
  class ReadTagRes {
    TagType  mType;
    int      mNextChar;
  };
  
  boolean isValidFirstChar(int cInt) {
    if (cInt < 0) return false;
    char c = (char)cInt;
    // See rules http://www.w3.org/TR/REC-xml/#NT-NameStartChar
    return  
       (c >= 'a' && c <='z') ||
       (c >= 'A' && c <='Z') ||
        c == ':' || c == '_' ||
       (cInt >= 0xC0 && cInt <= 0xD6) ||
       (cInt >= 0xD8 && cInt <= 0xF6)  || 
       (cInt >= 0xF8 && cInt <= 0x2FF) || 
       (cInt >= 0x370 && cInt <= 0x37D) || 
       (cInt >= 0x37F && cInt <= 0x1FFF) || 
       (cInt >= 0x200C && cInt <= 0x200D) || 
       (cInt >= 0x2070 && cInt <= 0x218F) || 
       (cInt >= 0x2C00 && cInt <= 0x2FEF) || 
       (cInt >= 0x3001 && cInt <= 0xD7FF) || 
       (cInt >= 0xF900 && cInt <= 0xFDCF) || 
       (cInt >= 0xFDF0 && cInt <= 0xFFFD) || 
       (cInt >= 0x10000 && cInt <= 0xEFFFF);       
  }
  
  boolean isValidOtherChar(int cInt) {
    if (cInt < 0) return false;
    char c = (char)cInt;
    return isValidFirstChar(c) ||
            c == '-' || 
            c == '.' || 
            cInt == 0xB7 ||
            (cInt >= 0x0300 && cInt <= 0x036F) || 
            (cInt >= 0x203F && cInt <= 0x2040) ;
  }
  
  /**
   * This function attempts to read the longest sequence that is a valid tag name,
   * i.e., it terminates on seeing a space, '>', etc...
   * 
   * @param rtag    a structure to hold both the read tag name and the next char beyond the name.            
   * @return    true if at least one valid character was read and false otherwise
   * @throws IOException 
   */
  private boolean readTagName(ReadTagRes rtag) throws IOException {
    rtag.mType = TagType.TAG_REG_OPEN;  
    rtag.mNextChar = -1;
    int  c = readChar();
    char cFirst = (char)c;
    if (cFirst == '/') {
      c = readChar();
      rtag.mType = TagType.TAG_REG_CLOSE;
    } else if (cFirst == '!' || cFirst == '?' ) {
      // CDATA tag http://www.w3.org/TR/REC-xml/#sec-cdata-sect
      // preamble or 
      mTagBuff.setLength(0); // reset buffer content
            
      for(int i=0 ; true; ++i) {
        if (
            (i < CDATA_TAG.length()     && c == CDATA_TAG.charAt(i)) ||
            (i < DOCTYPE_TAG.length()   && c == DOCTYPE_TAG.charAt(i)) ||
            (i < PREAMBLE_TAG.length()  && c == PREAMBLE_TAG.charAt(i)) 
            )
          mTagBuff.append((char)c);
        else {
          rtag.mNextChar = (char)c;
          break;
        }
        c = readChar();
      }
      if (cFirst == '!') {
        if (stringEqualsBuffer(CDATA_TAG, mTagBuff)) {
          rtag.mType = TagType.TAG_CDATA;
          return true;
        }
        if (stringEqualsBuffer(DOCTYPE_TAG, mTagBuff)) {
          rtag.mType = TagType.TAG_DOCTYPE;
          return true;
        }
        return false;
      }
      if (cFirst == '?') {
        if (!stringEqualsBuffer(PREAMBLE_TAG, mTagBuff)) return false;
        rtag.mType = TagType.TAG_PREAMBLE;
        return true;      
      }
      
      return false;
    }
    if (!isValidFirstChar(c)) 
      return false; // -1 will not result in a valid character

    mTagBuff.setLength(0); // reset buffer content
    mTagBuff.append((char)c);
    while (true) {
      c = readChar();
      if (isValidOtherChar(c)) 
        mTagBuff.append((char)c);
      else {
        rtag.mNextChar = (char)c;
        break;
      }
    }

    return true;
  }
  
  /**
   * 
   * Try to read the next element.
   * 
   * @return the text of the element or the empty string, when no further element is found.
   * @throws Exception
   */
  public String readNext() throws Exception {
    char cbuff1[] = new char[1];
    
    ReadTagRes  rtag = new ReadTagRes();
    
    while(true) {
      /* 
       * Syntax rules for XML tags:
       * http://www.w3.org/TR/REC-xml/#sec-starttags
       * 1) There can be no spaces between '<' and the tag name
       * 2) There can be no spaces between '</' and the tag name
       * 3) There can be no space between '<' and the following '/'
       * 4) There can be no space between '/' and the following '>' 
       */
      while(true) {
        mCurrBuff.setLength(0); // Reset buffer contents
        
        
        if (!scanToChar(OPEN_BRACKET_CHAR, null)) return "";
        
        mCurrBuff.append(OPEN_BRACKET_CHAR);
        
        int prevQty = mReadQty;
        if (!readTagName(rtag)) {
          throw new 
          Exception(String.format("Wrong XML format, didn't find the tag name/CDATA " +
              " after char # %d, current buffer: '%s'", 
              prevQty, mCurrBuff));                        
        }
        
        // All CDATA outside of the enclosing tag should be ignored
        if (rtag.mType == TagType.TAG_CDATA) {
          prevQty = mReadQty;
          if (!scanToEndCdata(mCurrBuff)) { // we memorize what we scan in case we need to report an error
            throw new 
            Exception(String.format("Wrong XML format, didn't find the end of CDATA " +
                " after char # %d, current buffer: '%s'", 
                prevQty, mCurrBuff));      
          }           
          continue; // discard what we have seen so far
        }
        // All DATA and the whole preamble should be ignored
        if (rtag.mType == TagType.TAG_DOCTYPE || rtag.mType == TagType.TAG_PREAMBLE) {
          prevQty = mReadQty;
          if (!scanToClosingBalanced(mCurrBuff)) { // we memorize what we scan in case we need to report an error
            throw new 
            Exception(String.format("Wrong XML format, didn't find the end of " +
                rtag.mType + 
                " after char # %d, current buffer: '%s'", 
                prevQty, mCurrBuff));      
          }            
          continue; // discard what we have seen so far         
        }        
        
        if (rtag.mType == TagType.TAG_REG_CLOSE) mCurrBuff.append(SLASH_CHAR);
        mCurrBuff.append(mTagBuff);
        if (rtag.mNextChar != -1) mCurrBuff.append((char)rtag.mNextChar);
        
       

        if (rtag.mType != TagType.TAG_REG_CLOSE && // No closing tags here!!! 
            stringEqualsBuffer(mEnclTagName, mTagBuff)) {      
          /*
           *  Here we found the starting portion of the tag.
           *  It may be followed by: 
           *  1) '>'  (opening tag)
           *  2) a whitespace (opening or self-closed tag)
           *  3) "/>" (self-closed tag)
           *  4) other alpha-numerical chars, i.e., be a prefix of another tag
           *  
           */
          int cNext = rtag.mNextChar;
          if (cNext == CLOSE_BRACKET_CHAR) {
            // Need to scan to find the closing tag
            break;
          } else if (Character.isWhitespace(cNext)) {
            prevQty = mReadQty;
            if (!scanToChar(CLOSE_BRACKET_CHAR, 
                            mCurrBuff // in this scan we do want to memorize the content
                             )) {
              throw new Exception(String.format(
                            "Wrong XML format, expecting '%c' after char # %d, current buffer: '%s' ", 
                             CLOSE_BRACKET_CHAR, prevQty, mCurrBuff));            
            }            
            
            // currBuff.length() should be at least 2
            // Let's check if we a slash before the closing bracket
            mCurrBuff.getChars(mCurrBuff.length() - 2, mCurrBuff.length() - 1, 
                              cbuff1, 0);
            // An empty element (self-closing tag), can't be separated by space from '>'
            if (cbuff1[0] == SLASH_CHAR) return mCurrBuff.toString();
            // Need to scan to find the closing tag
            break;
          } else if (cNext == SLASH_CHAR) {
            int cNextNext = readChar();
            if (-1 == cNextNext) {
              throw new 
              Exception(String.format("Wrong XML format, reached EOF " +
                                       "while expecting '%c' after char # %d, current buffer: '%s'", 
                                       CLOSE_BRACKET_CHAR, mReadQty, mCurrBuff));                              
            }
            if (cNextNext != CLOSE_BRACKET_CHAR) {
              throw new 
              Exception(String.format("Wrong XML format, reached EOF " +
                                       "expecting '%c' after char # %d, current buffer: '%s'", 
                                       CLOSE_BRACKET_CHAR, mReadQty, mCurrBuff));            
            }
            // An empty element (self-closing tag)
            mCurrBuff.append(CLOSE_BRACKET_CHAR); 
            return mCurrBuff.toString();
          }
        }
        /*
         *  Not our tag of interest, scan to the next closing bracket,
         *  but only if we haven't seen this closing bracket already.
         *  The content of the buffer will be erased in the beginning
         *  of the loop. 
         */
        if (rtag.mNextChar != CLOSE_BRACKET_CHAR) {
          prevQty = mReadQty;
          if (!scanToChar(CLOSE_BRACKET_CHAR, null)) {
            throw new Exception(String.format(
                          "Wrong XML format, expecting '%c' after char # %d, current buffer: '%s'", 
                           CLOSE_BRACKET_CHAR, prevQty, mCurrBuff));            
          }
        }
      }

      while (true) {
        // Let's find the closing tag
        int prevQty = mReadQty;
        if (!scanToChar(OPEN_BRACKET_CHAR, 
                        mCurrBuff // in this scan we do want to memorize the content
                        )) {
          throw new Exception(String.format(
                        "Wrong XML format, expecting '%c' after char # %d, current buffer: '%s'", 
                        OPEN_BRACKET_CHAR, prevQty, mCurrBuff));            
        }        
        prevQty = mReadQty;
        if (!readTagName(rtag)) {
          throw new 
          Exception(String.format("Wrong XML format, didn't find the tag name/CDATA " +
              " after char # %d, current buffer: '%s'", 
              prevQty, mCurrBuff));                        
        }
        
        if (rtag.mType == TagType.TAG_DOCTYPE ||
            rtag.mType == TagType.TAG_PREAMBLE) {
          throw new 
          Exception(String.format("Wrong XML format, unexpected tag '" +
              mTagBuff + "'" +
              " after char # %d, current buffer: '%s'", 
              prevQty, mCurrBuff));                                  
        }
        
        if (rtag.mType == TagType.TAG_CDATA) {
          mCurrBuff.append(mTagBuff);
          int cNext = rtag.mNextChar;
          if (cNext != -1) mCurrBuff.append((char)cNext);
         
          prevQty = mReadQty;
          if (!scanToEndCdata(mCurrBuff)) {
            throw new 
            Exception(String.format("Wrong XML format, didn't find the end of CDATA " +
                " after char # %d, current buffer: '%s'", 
                prevQty, mCurrBuff));      
          }
          /*
           * If we read the CDATA block, let's memorize contents
           * and continue searching for the terminating enclosing tag.
           */
          continue;
        }        
        
        if (rtag.mType == TagType.TAG_REG_CLOSE) mCurrBuff.append(SLASH_CHAR);
        mCurrBuff.append(mTagBuff);
        int cNext = rtag.mNextChar;
        if (cNext != -1) mCurrBuff.append((char)cNext);
        
        if (rtag.mType != TagType.TAG_REG_CLOSE) continue; // Ignore if this is not a closing tag
               
        if (stringEqualsBuffer(mEnclTagName, mTagBuff)) {

          while (Character.isWhitespace(cNext)) { // Skip possible whitespaces
            prevQty = mReadQty;            
            cNext = readChar();
            
            if (-1 == cNext) {
              throw new 
              Exception(String.format("Wrong XML format, reached EOF " +
                                       "while expecting '%c' after char # %d, current buffer: '%s'", 
                                       CLOSE_BRACKET_CHAR, mReadQty,mCurrBuff));                              
            }
            mCurrBuff.append((char)cNext);
          } 
          if (CLOSE_BRACKET_CHAR == cNext) return mCurrBuff.toString();
          // uugh, not exactly the closing tag, but a tag that has the same prefix
        }
      }
    }   
  }
  
  /**
   * Allocation-less comparison between the string and the buffer.
   * 
   * @param tag     string to compare.
   * @param sb      buffer contents to compare.
   * @return        true if the buffer contents are equal to the string contents.
   */
  private boolean stringEqualsBuffer(String tag, StringBuffer sb) {
    int len = tag.length();
    if (len != sb.length()) return false;
    for (int i = 0; i < len; ++i) {
      if (sb.charAt(i) != tag.charAt(i)) return false;
    }
    return true;
  }

  /**
   * Scans the input till "]]>" that terminates the opening CDATA tag.
   * 
   * @param currBuff       the buffer to store the input
   * @return               flag: did we find the end of CTAG block?
   * @throws IOException 
   */
  private boolean scanToEndCdata(StringBuffer currBuff) throws IOException {
    // CDATA tag http://www.w3.org/TR/REC-xml/#sec-cdata-sect
    int c1 = readChar();
    if (-1 == c1) return false;
    if (currBuff != null) currBuff.append((char)c1);
    int c2 = readChar();
    if (-1 == c2) return false;
    if (currBuff != null) currBuff.append((char)c2);
    while (true) {
      int c3 = readChar();
      if (c3 == -1) return false;
      if (currBuff != null) currBuff.append((char)c3);
      if (']' == c1 && c1 == c2 && c3 == '>') {
        return true;
      }
      c1 = c2;
      c2 = c3;
    }
  }
  
  /**
   * Scans the input until the closing bracket is found; permits
   * balanced angled brackets inside, e.g., as in the !DOCUMEN.
   * 
   * @param currBuff       the buffer to store the input
   * @return               did we find the ending 
   * @throws IOException
   */
  private boolean scanToClosingBalanced(StringBuffer currBuff) throws IOException {
    int qty = 1;
    while (true) {
      int c = readChar();
      if (-1 == c) return false;
      if (currBuff != null) currBuff.append((char)c);
      if (c == '<') ++qty;
      if (c == '>') --qty;
      if (0 == qty) return true;
    }
  }

  /**
   * @return a read character or -1, if the EOF is reached.
   * @throws IOException
   */
  private int readChar() throws IOException {
    int c = mInput.read();
    if (c != -1) ++mReadQty;
    return c;
  }
  
  /**
   * Reads all the characters until the specified one (inclusive);
   * If necessary, stores these characters in a buffer. 
   * 
   * @param expecting       a character to expect
   * @param currBuff        the buffer to save read char or NULL (to discard all read ones)
   * @return
   * @throws IOException
   */
  private boolean scanToChar(char expectedChar,
                             StringBuffer currBuff) throws IOException {
    int c = -1;
    do {
      c = readChar();
      if (-1 == c) return false;
      if (null != currBuff) currBuff.append((char)c);
    }  while (c != expectedChar);

    return true;
  }

  StringBuffer              mCurrBuff = new StringBuffer();
  StringBuffer              mTagBuff  = new StringBuffer();
  private int               mReadQty = 0;
  private BufferedReader    mInput;
  private String            mEnclTagName;


  public void close() throws IOException {
    mInput.close();    
  }
}
