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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.*;
import java.util.*;

import no.uib.cipr.matrix.DenseVector;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.utils.EvalUtils;


class QueryAppImpl extends BaseQueryApp {      
  @Override
  void addOptions() {
    boolean onlyLucene    = false;
    boolean multNumRetr   = true;
    boolean useQRELs      = true;
    boolean useThreadQty  = true;    
    addCandGenOpts(onlyLucene, 
                   multNumRetr,
                   useQRELs,
                   useThreadQty);
    
    addResourceOpts();
    
    boolean useIntermModel = true, useFinalModel = true;
    addLetorOpts(useIntermModel, useFinalModel);
    
    mOptions.addOption(CommonParams.TREC_STYLE_OUT_FILE_PARAM, null, true,  CommonParams.TREC_STYLE_OUT_FILE_DESC);    
  }

  @Override
  void procCustomOptions() {
    mOutPrefix = mCmd.getOptionValue(CommonParams.TREC_STYLE_OUT_FILE_PARAM);
    if (mOutPrefix == null) 
      showUsageSpecify(CommonParams.TREC_STYLE_OUT_FILE_DESC);
  }

  @Override
  void init() throws IOException {
    for (int numRet : mNumRetArr) {
      String outFile = outFileName(mOutPrefix, numRet);
      mhOutFiles.put(numRet, new BufferedWriter(new FileWriter(new File(outFile))));
    }    
  }

  @Override
  void fin() throws IOException {
    for (Map.Entry<Integer, BufferedWriter> e: mhOutFiles.entrySet()) {
      e.getValue().close();
    }
  }

  @Override
  void procResults(String runId, 
                   String queryId, 
                   Map<String, String> docFields, 
                   CandidateEntry[] scoredDocs, int numRet, Map<String, DenseVector> docFeats) throws IOException {
    BufferedWriter trecOut = mhOutFiles.get(numRet);
    if (null == trecOut) 
      throw new RuntimeException("Bug, output file is not init. for numRet=" + numRet);
    EvalUtils.saveTrecResults(queryId, 
                              scoredDocs,
                              trecOut, 
                              runId, 
                              scoredDocs.length);
  }

  String outFileName(String filePrefix, int numRet) {
    return filePrefix + "_" + numRet;
  }
  
  private final HashMap<Integer, BufferedWriter>  mhOutFiles = new HashMap<Integer, BufferedWriter>();
  private String mOutPrefix;
}

public class QueryAppMultThread {
  
  public static void main(String[] args) {
    try {
      (new QueryAppImpl()).run("Query application", args);
    } catch(Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
  }  
  
}
