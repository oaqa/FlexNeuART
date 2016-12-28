/*
 *  Copyright 2016 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.squad;

import java.io.*;

import com.google.gson.*;

import edu.cmu.lti.oaqa.annographix.util.CompressUtils;

public class SQuADReader {

  public SQuADReader(String inputFile) throws IOException {
    final BufferedReader  input 
        = new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(inputFile)));
    
    StringBuffer sb = new StringBuffer();
    String s;
    while ((s=input.readLine()) != null) {
      sb.append(s);
      sb.append('\n');
    }
    input.close();
    
    Gson gson = new Gson();
    mData = gson.fromJson(sb.toString(), SQuADData.class);
  }

  public static void main(String[] args) throws IOException {
    String fileName = args[0];
    SQuADReader  r = new SQuADReader(fileName);
    
    int qty = 0;
    int parQty = 0;
    for (SQuADEntry e : r.mData.data) {
      for (SQuADParagraph p : e.paragraphs) {
       qty += p.qas.length;
       ++parQty;
       for (SQuADQuestionAnswers qas : p.qas) {
          //System.out.println(qas.question);
        }
      }
    }
    
    System.out.println("File name:            " + fileName);
    System.out.println("Version:              " + r.mData.version);    
    System.out.println("Number of pages:      " + r.mData.data.length);
    System.out.println("Number of paragarphs: " + parQty);
    System.out.println("Number of questions:  " + qty);
  }
  
  public final SQuADData   mData;
}
