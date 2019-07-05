/*
 *  Copyright 2017 Carnegie Mellon University
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

import org.apache.commons.cli.*;


import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.letor.EvalUtils;

import java.util.*;

/**
 * Compute k-NN/overlap statistics from TREC_RUNS.
 * 
 * @author Leonid Boytsov
 *
 */

public class ComputeKNNStatFromTrecRuns {
  private static final String TREC_RUN_EXACT_PARAM   = "trec_run_exact";
  private static final String TREC_RUN_APPROX_PARAM  = "trec_run_approx";
  private static final String QREL_FILE_PARAM        = CommonParams.QREL_FILE_PARAM;

  static void Usage(String err, Options opt) {
    System.err.println("Error: " + err);
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "ComputeKNNStatFromTrecRuns", opt);      
    System.exit(1);
  }  
  
  public static void main(String [] args) {
    Options options = new Options();
    
    options.addOption(TREC_RUN_EXACT_PARAM,  null, true, "exact trec_run file (can be gz or bz2 compressed)");   
    options.addOption(TREC_RUN_APPROX_PARAM, null, true, "approximate trec_run file (can be gz or bz2 compressed)");
    options.addOption(QREL_FILE_PARAM,       null, true, "qrel files");
    
    CommandLineParser parser = new org.apache.commons.cli.GnuParser();
    
    try {
      CommandLine cmd = parser.parse(options, args);
    
      String exactFileName = cmd.getOptionValue(TREC_RUN_EXACT_PARAM);      
      if (null == exactFileName) Usage("Specify: " + TREC_RUN_EXACT_PARAM, options);
      String approxFileName = cmd.getOptionValue(TREC_RUN_APPROX_PARAM);
      if (null == approxFileName) Usage("Specify: " + TREC_RUN_APPROX_PARAM, options);
      String qrelFileName = cmd.getOptionValue(QREL_FILE_PARAM);
      if (null == qrelFileName) Usage("Specify " + QREL_FILE_PARAM, options);
      
      HashMap<String, HashMap<String, Integer>> qrels = EvalUtils.readQrelEntries(qrelFileName);
      HashMap<String, ArrayList<CandidateEntry>> exactRuns = EvalUtils.readTrecResults(exactFileName);
      HashMap<String, ArrayList<CandidateEntry>> approxRuns = EvalUtils.readTrecResults(approxFileName);
      
      float qty = 0, totOverlap = 0, recall1 = 0;
      
      for (String topicId :  exactRuns.keySet()) {
        float r = computeRecall(exactRuns.get(topicId), approxRuns.get(topicId));
        recall1 += computeRecall1(exactRuns.get(topicId), approxRuns.get(topicId));
        System.out.println(topicId + " " + r);
        totOverlap += r;
        ++qty;        
      }
      
      System.out.println("=========================");
      System.out.println(String.format("# of topics %d, recall %f recall@1 %f", (int)qty, totOverlap/qty, recall1/qty));
      
    } catch (ParseException e) {
      Usage("Cannot parse arguments", options);
    } catch(Exception e) {
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    } 
    
  }

  private static float computeRecall1(ArrayList<CandidateEntry> exactRun,
                                      ArrayList<CandidateEntry> approxRun) {
    if (approxRun == null) return 0;
    if (approxRun.isEmpty()) return exactRun.isEmpty() ? 1 : 0;
    String did = exactRun.get(0).mDocId;
    for (CandidateEntry e : approxRun) {
      if (e.mDocId.equals(did)) return 1;
    }
    return 0;
}  
  
  private static float computeRecall(ArrayList<CandidateEntry> exactRun,
                                     ArrayList<CandidateEntry> approxRun) {
    if (approxRun == null) return 0;
    if (approxRun.isEmpty()) return exactRun.isEmpty() ? 1 : 0;
    HashSet<String> ekey = new HashSet<String>(), akey = new HashSet<String>();
    for (CandidateEntry e : exactRun) ekey.add(e.mDocId);
    for (CandidateEntry e : approxRun) akey.add(e.mDocId);
    
    float origQty = akey.size();
    
    akey.removeAll(ekey);
    
    return (origQty - akey.size())/origQty;
  }
}
