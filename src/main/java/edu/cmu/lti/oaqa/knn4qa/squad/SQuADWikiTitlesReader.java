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

import java.util.*;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;

public class SQuADWikiTitlesReader {
  public final static String[] mFiles = {
    "dev-v1.1.json.gz",
    "train-v1.1-split_dev1.gz",
    "train-v1.1-split_dev2.gz",
    "train-v1.1-split_train.gz",
    "train-v1.1-split_tran.gz"
  };
  
  public SQuADWikiTitlesReader(String inputDir) throws Exception {
    for (String fn : mFiles) {
      SQuADReader r = new SQuADReader(inputDir + "/" + fn);
      
      if (r.mData.data == null || r.mData.data.length == 0)
        throw new Exception("No data found in the file: '" + inputDir + "'");
      
      for (SQuADEntry e : r.mData.data) {
        String title = decodeTitle(e.title);
        if (!mhTitles.contains(title)) 
          mhTitles.add(title);
      }
    }
  }
  
  private String decodeTitle(String title) throws UnsupportedEncodingException {
    return URLDecoder.decode(title, "utf8").replace('_', ' ');
  }

  public static void main(String[] args) throws Exception {
    SQuADWikiTitlesReader tr = new SQuADWikiTitlesReader(args[0]);
    for (String t : tr.mhTitles) {
      System.out.println(t);
    }
  }
  
  public final Set<String> mhTitles = new TreeSet<String>();

}
