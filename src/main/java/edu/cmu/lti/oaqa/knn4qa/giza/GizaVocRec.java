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
package edu.cmu.lti.oaqa.knn4qa.giza;

import java.io.BufferedWriter;
import java.io.IOException;


public class GizaVocRec {
  public final String  mWord;
  public final int     mId;
  public final int     mQty;
  
  public GizaVocRec(String line) throws Exception {
    String parts[] = line.trim().split("\\s+");
    if (parts.length != 3) {
      throw new Exception(
          String.format("Wrong format of line '%s', got %d fields instead of three.",
                        line, parts.length));
    }
    mWord = parts[1];
    
    try {
      mId  = Integer.parseInt(parts[0]);
      mQty = Integer.parseInt(parts[2]);
    } catch (NumberFormatException e) {
      throw new Exception(
          String.format("Wrong format of line '%s', either ID or the quantity field doesn't contain a proper integer.",
                        line));      
    }
  }
  
  public void save(BufferedWriter fout) throws IOException {
    fout.append(String.format("%d %s %d", mId, mWord, mQty));
    fout.newLine();
  }
  
  
  public GizaVocRec(String word, int id, int qty) {
    mWord = word;
    mId = id;
    mQty = qty;
  }
}
