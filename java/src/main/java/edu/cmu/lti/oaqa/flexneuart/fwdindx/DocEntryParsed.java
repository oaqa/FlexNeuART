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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

import edu.cmu.lti.oaqa.flexneuart.utils.Const;

/**
 * One document entry in the forward in-memory index. We regret this 
 * class was not conveniently split into two classes. However, such
 * a refactoring will break quite a few of LOCs, so we at least postpone it.
 * Furthermore, saving to/restoring from a text format should be retired.
 * 
 * @author Leonid Boytsov
 *
 */
public class DocEntryParsed {

  public static final String DOCLEN_QTY_PREFIX = "@ ";
  public static final int[] EMPTY_INT_ARRAY = new int[0];
  

  public final int mWordIds[]; // unique word IDs
  public final int mQtys[];    // # of word occurrences corresponding to {@link mWordIds} 
  // A sequence of word IDs (as they appear in the corresponding field), this array CAN BE NULL.
  public final int mWordIdSeq[]; 
  public final int mDocLen; // document length in # of words
	
  public DocEntryParsed(int uniqQty, int [] wordIdSeq, boolean bStoreWordIdSeq) {
    mWordIds = new int [uniqQty];
    mQtys    = new int [uniqQty];
    if (bStoreWordIdSeq)
      mWordIdSeq = wordIdSeq;
    else
      mWordIdSeq = EMPTY_INT_ARRAY;
    mDocLen = wordIdSeq.length;
  }

  public DocEntryParsed(int uniqQty, int [] wordIdSeq, int docLen) {
    mWordIds = new int [uniqQty];
    mQtys    = new int [uniqQty];
    // We do not treat empty arrays as null anymore
    if (wordIdSeq != null) {
      mWordIdSeq = wordIdSeq;
      if (wordIdSeq.length != docLen) {
      	throw new RuntimeException("Bug: different lengths docLen=" + docLen + " wordIdSeq.length=" + wordIdSeq.length);
      }
    } else {
      mWordIdSeq = EMPTY_INT_ARRAY;
    }
    mDocLen = docLen;
  }
  
  public DocEntryParsed(ArrayList<Integer> wordIds, 
                  		ArrayList<Integer> wordQtys,
                  		int                docLen) {
    mWordIds = new int [wordIds.size()];
    mQtys    = new int [wordIds.size()];
    
    for (int i = 0; i < wordIds.size(); ++i) {
      mWordIds[i] = wordIds.get(i);
      mQtys[i] = wordQtys.get(i);
    }
    
    mWordIdSeq = EMPTY_INT_ARRAY;
    mDocLen = docLen;
  }

  
  public DocEntryParsed(ArrayList<Integer> wordIds, 
                  ArrayList<Integer> wordQtys,
                  ArrayList<Integer> wordIdSeq,
                  boolean bStoreWordIdSeq) throws Exception {
    if (wordIds.size() != wordQtys.size()) {
      throw new Exception("Bug: the number of word IDs is not equal to the number of word quantities.");
    }
    mWordIds = new int [wordIds.size()];
    mQtys    = new int [wordIds.size()];
    
    for (int i = 0; i < wordIds.size(); ++i) {
      mWordIds[i] = wordIds.get(i);
      mQtys[i] = wordQtys.get(i);
    }
    
    if (bStoreWordIdSeq) {
    	// Let's keep empty sequences as empty arrays rather than null pointers
      mWordIdSeq = new int [wordIdSeq.size()];
      for (int k = 0; k < wordIdSeq.size(); ++k) {
        mWordIdSeq[k] = wordIdSeq.get(k);
      }
    } else {
      mWordIdSeq = EMPTY_INT_ARRAY;
    }
    mDocLen = wordIdSeq.size();
  }
  
  /**
   * Generate a string representation of a document entry:
   * it doesn't add a trailing newline!
   */
  public String toString() {
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < mWordIds.length; ++i) {
      if (i > 0) sb.append('\t');
      sb.append(String.format("%d:%d",mWordIds[i] , mQtys[i]));
    }
    sb.append('\n');
    // Do not create an empty string for the empty sequence,
    // add a special marker with the document length
    if (mWordIdSeq != null && mWordIdSeq.length > 0) {
      for (int i = 0; i < mWordIdSeq.length; ++i) {
        if (i > 0) sb.append(' ');
        sb.append(mWordIdSeq[i]);
      }
    } else {
      sb.append(DOCLEN_QTY_PREFIX + mDocLen);
    }
    
    return sb.toString();
  }
  
  /**
   * Parse a textual DocEntry representation.
   */
  public static DocEntryParsed fromString(String text) throws Exception {
    int nli = text.indexOf('\n');
    if (nli < 0) {
      throw new Exception("Invalid document entry, no newline found in:\n" + text);
    }
    String line1 = text.substring(0, nli);

    ArrayList<Integer> wordIds = new ArrayList<Integer>(), wordQtys = new ArrayList<Integer>();
    
    // Some lines can be empty
    if (!line1.isEmpty()) {
      String[] parts = line1.split("\\s+");
      int k = 0;
      for (String s: parts) {
        String [] parts2 = s.split(":");
        ++k;
        if (parts2.length != 2) {
          throw new Exception(
              String.format(
                  "Wrong format: expecting two colon-separated numbers in the substring:" +
                  "'%s' (part # %d)", s, k));          
        }
        
        try {
          int wordId = Integer.parseInt(parts2[0]);
          int wordQty = Integer.parseInt(parts2[1]);
          
          wordIds.add(wordId);
          wordQtys.add(wordQty);
        } catch (NumberFormatException e) {
          throw new Exception(
              String.format(
                      "Invalid document entry format (an ID or count isn't integer), in the substring '%s' (part #%d)",
                      s, k));
        }          
      }
    }
    
    
    String line2 = text.substring(nli + 1);
    
    if (line2.startsWith(DOCLEN_QTY_PREFIX)) {
      final int docLen;
      try {
        docLen = Integer.parseInt(line2.substring(DOCLEN_QTY_PREFIX.length()));
      } catch (NumberFormatException e) {
        throw new Exception("Document length isn't integer");               
      }
      return new DocEntryParsed(wordIds, wordQtys, docLen); 
    } else {
      ArrayList<Integer> wordIdSeq = new ArrayList<Integer>();
      
      if (!line2.isEmpty()) {
        String [] parts = line2.split("\\s+");
        int k = 0;
        for (String s: parts) {
          ++k;
          try {
            int wordId = Integer.parseInt(s);
            wordIdSeq.add(wordId);
          } catch (NumberFormatException e) {
            throw new Exception(
                String.format(
                        "Invalid document entry format (word ID isn't integer), in the substring '%s' (part #%d)",
                        s, k));
          }          
        }
      }
      
      return new DocEntryParsed(wordIds, wordQtys, wordIdSeq, true);
    }       
  }

  
  /**
   * Generate a binary representation of a document entry.
   */
  public byte[] toBinary() {
    
    if (mWordIdSeq.length > 0 &&
        mWordIdSeq.length != mDocLen) {
      throw new RuntimeException("Bug: different lengths docLen=" + mDocLen + 
                                  " wordIdSeq.length=" + mWordIdSeq.length);
    }
    
  	int totalSize = 4 + // # of (word id, qty) pairs 
  					4 + // # the length of a word sequence
  					mWordIds.length * 8 + // id + qty
  					mWordIdSeq.length * 4;
  	ByteBuffer out = ByteBuffer.allocate(totalSize);
  	out.order(Const.BYTE_ORDER);
  	out.putInt(mWordIds.length);
  	// minus indicates there will be no word id sequence
  	out.putInt(mWordIdSeq.length > 0 ? mDocLen : -mDocLen);
  	for (int i = 0; i < mWordIds.length; ++i) {
      out.putInt(mWordIds[i]);
      out.putInt(mQtys[i]);
    }

    for (int i = 0; i < mWordIdSeq.length; ++i) {
    	out.putInt(mWordIdSeq[i]);
    }
  	
  	return out.array();
  }
  
  /**
   * Parse a binary DocEntry representation. We try to make this function as efficient
   * as possible so that it doesn't make extra allocations.
   */
  public static DocEntryParsed fromBinary(byte[] bin) throws Exception {
  	ByteBuffer in = ByteBuffer.wrap(bin);
  	in.order(Const.BYTE_ORDER);
  	int uniqueQty = in.getInt();
  	int docLen = in.getInt();
  	int wordIdSeq[] = null;

  	if (docLen < 0) {
  	  docLen = -docLen;
  	} else if (docLen > 0) {
  	   // At this point, unfortunately, we do not know 
  	  wordIdSeq = new int[docLen];
  	} else {
  	  // If docLen == 0, the constructor will init. mWordIdSeq to an empty array
  	}
  	
  	DocEntryParsed res = new DocEntryParsed(uniqueQty, wordIdSeq, docLen);
    // After doc entry is created we need to fill out the following
  	// 1) doc ids and qtys
  	// 2) optionally retrieve a sequence of word ids
  	for (int i = 0; i < uniqueQty; i++) {
  		res.mWordIds[i] = in.getInt();
  		res.mQtys[i] = in.getInt();
  	}
  	if (res.mWordIdSeq == null) {
  	  throw new Exception("Bug: we should not get null here, as a constructor should use the empty array instead!");
  	}
	for (int i = 0; i < res.mWordIdSeq.length; ++i) {
	  res.mWordIdSeq[i] = in.getInt();
	}
  	
  	return res;
  	
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    DocEntryParsed other = (DocEntryParsed) obj;
    if (mDocLen != other.mDocLen)
      return false;
    if (!Arrays.equals(mQtys, other.mQtys))
      return false;
    if (!Arrays.equals(mWordIdSeq, other.mWordIdSeq))
      return false;
    if (!Arrays.equals(mWordIds, other.mWordIds))
      return false;
    return true;
  }

}
