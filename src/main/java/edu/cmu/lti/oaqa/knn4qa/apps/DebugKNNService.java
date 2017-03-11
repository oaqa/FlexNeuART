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
package edu.cmu.lti.oaqa.knn4qa.apps;

import java.util.*;

import no.uib.cipr.matrix.DenseVector;

import com.google.common.base.Splitter;


import edu.cmu.lti.oaqa.knn4qa.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.knn4qa.cand_providers.NmslibQueryGenerator;

/* begin: Imports related to a KNN-service */ 
import org.apache.thrift.transport.*;

import edu.cmu.lti.oaqa.similarity.*;

import org.apache.thrift.protocol.*;
/* end: Imports related to a KNN-service */

class DebugKNNServicImpl extends BaseQueryApp {
  public static final String LUCENE_INDEX_LOCATION_DESC = "Location of a Lucene index";
  
  private static final int MAX_DIGITS_TO_COMPARE = 4;
  
  int mQtyPotentMismatch = 0;
  int mQtyComp = 0;
  
  String              mKnnServiceURL;
  TTransport          mKnnServiceTransp;
  QueryService.Client mKnnServiceClient;
  
  
  NmslibQueryGenerator mQueryGen;
  
  @Override
  void addOptions() {
    boolean onlyLucene    = true;
    boolean multNumRetr   = false;
    boolean useQRELs      = false;
    boolean useThreadQty  = false;    
    addCandGenOpts(onlyLucene, 
                   multNumRetr,
                   useQRELs,
                   useThreadQty);
    
    boolean useHigHorderModels = false;
    addResourceOpts(useHigHorderModels);
    
    boolean useIntermModel = true, useFinalModel = false;
    addLetorOpts(useIntermModel, useFinalModel);
    
    mOptions.addOption(CommonParams.KNN_SERVICE_PARAM,         null, true,  CommonParams.KNN_SERVICE_DESC);    
  }
  
  @Override
  void procCustomOptions()  {    
    mKnnServiceURL = mCmd.getOptionValue(CommonParams.KNN_SERVICE_PARAM);
    if (null == mKnnServiceURL) showUsageSpecify(CommonParams.KNN_SERVICE_DESC);      
  }
  
  
  @Override
  void init() throws Exception {
    if (null == mInMemExtrInterm)
      showUsageSpecify(CommonParams.EXTRACTOR_TYPE_INTERM_DESC);
    if (null == mNmslibFields) showUsageSpecify(CommonParams.NMSLIB_FIELDS_PARAM);
    
    mQueryGen = new NmslibQueryGenerator(mNmslibFields, mMemIndexPref, mInMemExtrInterm);
    
    Splitter splitOnColon = Splitter.on(':');
    
    String host = null;
    int    port = -1;
    
    int part = 0;
    for (String s : splitOnColon.split(mKnnServiceURL)) {
      if (0 == part) {
        host = s;
      } else if (1 == part) {
        try {
          port = Integer.parseInt(s);
        } catch (NumberFormatException e) {
          showUsage("Invalid port in the service address in '" + CommonParams.KNN_SERVICE_PARAM + "'");
        }
      } else {
        showUsage("Extra colon in the service address in '" + CommonParams.KNN_SERVICE_PARAM + "'");
      }
      ++part;
    }
    
    if (part != 2) {
      showUsage("Invalid format of the service address in '" + CommonParams.KNN_SERVICE_PARAM + "'");
    }
    
    mKnnServiceTransp = new TSocket(host, port);
    mKnnServiceTransp.open();
    mKnnServiceClient = new QueryService.Client(new TBinaryProtocol(mKnnServiceTransp));
  }

  @Override
  void fin() throws Exception {
    logger.info(String.format("# of comparisons %d, # of potential mismatches (see output before for details) %d", 
                              mQtyComp, mQtyPotentMismatch));
  }
    
  
  static boolean
  compareApprox(double a, double b, int digits) {
   double maxMod = Math.max(Math.abs(a), Math.abs(b));
   double scale =  Math.pow(10, digits);
   double lead  = Math.pow(10, Math.round(Math.log10(maxMod)));
  
   double minSign = Float.MIN_NORMAL * scale;
   // These guys are just too small for us to bother about their differences
   if (maxMod < minSign) return true;
   double delta = lead / scale;
   double  diff = Math.abs(a - b);
   return diff <= delta;
  }  
  
  @Override
  void procResults(String queryID, Map<String, String> docFields, 
                   CandidateEntry[] scoredDocs, int numRet, Map<String, DenseVector> docFeats) throws Exception {
    String queryObjStr = null;

    queryObjStr = mQueryGen.getStrObjForKNNService(docFields);

    logger.info("KNN Query string:");
    logger.info(queryObjStr);
    logger.info("==========================");

    for (CandidateEntry r : scoredDocs) {
      DenseVector feat = docFeats.get(r.mDocId);
      r.mScore = (float) feat.dot(mModelInterm);

      String docObjStr = mQueryGen.getStrObjForKNNService(r.mDocId);

      // Left queries
      double knnScore = -mKnnServiceClient.getDistance(docObjStr, queryObjStr);
      logger.info(String.format("docId=%s score=%f knn-Service dist=%f",
                                r.mDocId, r.mScore, knnScore));
      DenseVector v = docFeats.get(r.mDocId);
      for (int i = 0; i < v.size(); ++i) {
        if (i > 0) System.out.print(" ");
        System.out.print((i+1) + ":" + v.get(i));
      }
      System.out.println();

      String s1 = cutExtraDigits(r.mScore, MAX_DIGITS_TO_COMPARE);
      String s2 = cutExtraDigits(knnScore, MAX_DIGITS_TO_COMPARE);
      
      mQtyComp++;
      if (!compareApprox(r.mScore, knnScore, MAX_DIGITS_TO_COMPARE)) {
        mQtyPotentMismatch++;
        logger.info("Potential score mismatch!: " + s1 + " (Java) vs " + s2 + " (NMSLIB)");
      }   
    }
  } 

  
  /*
   * This function is not for display (because for integer numbers > 0, you will lose digits.
   * it is only for the purpose of approximate number comparison.
   */
  public static String cutExtraDigits(double val, int digitsToKeep) {
    String valStr = (val + "").toLowerCase().trim();
    String sign = "";
    if (valStr.startsWith("-")) {sign = "-";valStr=valStr.substring(1);}

    int pos = valStr.indexOf('e');
    String exp = "";
    String mant = valStr;
    if (pos >= 0) {
      exp = valStr.substring(pos);
      mant = valStr.substring(0, pos);
    }
    if (mant.length() > digitsToKeep + 1) mant = mant.substring(0, digitsToKeep + 1);
    return sign + mant + exp;
  }
}

public class DebugKNNService {    
  public static void main(String[] args) {
    try {
      (new DebugKNNServicImpl()).run("Debug KNN-service application", args);
    } catch(Exception e) {
      e.printStackTrace();
      System.err.println("Terminating due to an exception: " + e);
      System.exit(1);
    }     
  }  
}
