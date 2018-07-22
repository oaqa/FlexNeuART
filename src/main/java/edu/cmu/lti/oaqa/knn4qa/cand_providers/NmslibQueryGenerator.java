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

package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;

import edu.cmu.lti.oaqa.knn4qa.letor.FeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.letor.InMemIndexFeatureExtractor;
import edu.cmu.lti.oaqa.knn4qa.memdb.DocEntry;
import edu.cmu.lti.oaqa.knn4qa.memdb.InMemForwardIndex;

public class NmslibQueryGenerator {
  public NmslibQueryGenerator(String fieldNames[], 
                      String indexDir, 
                      InMemIndexFeatureExtractor ... donorExtractors) throws Exception {
    HashSet<String> hFieldNames = new HashSet<String>(Arrays.asList(fieldNames));
    HashSet<String> hActualFieldNames = new HashSet<String>(Arrays.asList(FeatureExtractor.mFieldNames));
    
    for (String fieldName : hFieldNames) {
      if (!hActualFieldNames.contains(fieldName)) {
        throw new Exception("Unknown field name: " + fieldName);
      }
    }
    
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) {
      int aliasOfId = FeatureExtractor.mAliasOfId[fieldId];
      int realFieldId = aliasOfId >= 0 ? aliasOfId : fieldId;
      String realFieldName = FeatureExtractor.mFieldNames[fieldId];
      if (hFieldNames.contains(realFieldName)) {
        mUseFiled[fieldId] = true;
        for (InMemIndexFeatureExtractor extr : donorExtractors)
          if (extr != null && mFieldIndex[realFieldId] == null) 
            mFieldIndex[realFieldId] = extr.getFieldIndex(realFieldId);
        if (null == mFieldIndex[realFieldId]) { // If the donor cannot be found, re-load index contents
          mFieldIndex[realFieldId] = 
              new InMemForwardIndex(FeatureExtractor.indexFileName(indexDir, FeatureExtractor.mFieldNames[realFieldId]));
        }
      }
    }
  }
    
  
  /**
   * Creates a string representation of the query that can be submitted to a
   * KNN-service.
   * 
   * @param docData   several pieces of input data, one is typically a bag-of-words query.
   * 
   * @return
   */
  public String getStrObjForKNNService(Map<String, String> docData) {   
    DocEntry[] docEntries = new DocEntry[FeatureExtractor.mFieldNames.length];
    
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) 
    if (mUseFiled[fieldId]) {
      int aliasOfId = FeatureExtractor.mAliasOfId[fieldId];
      int realFieldId = aliasOfId >= 0 ? aliasOfId : fieldId;

      if (mFieldIndex[realFieldId] != null) {
        String fieldQuery = docData.get(FeatureExtractor.mFieldNames[realFieldId]);
        docEntries[fieldId] = 
            mFieldIndex[realFieldId].createDocEntry(fieldQuery.split("\\s+"),
                                  true  /* True means we generate word ID sequence:
                                   * in the case of queries, there's never a harm in doing so.
                                   * If word ID sequence is not used, it will be used only to compute the document length. */              
                                  );    
        }
    }
    
    return getStrObjForKNNService(docEntries);
  }
  
  public String getStrObjForKNNService(String docId) {
    DocEntry[] docEntries = new DocEntry[FeatureExtractor.mFieldNames.length];
    
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) 
    if (mUseFiled[fieldId]) {
      int aliasOfId = FeatureExtractor.mAliasOfId[fieldId];
      int realFieldId = aliasOfId >= 0 ? aliasOfId : fieldId;

      if (mFieldIndex[realFieldId] != null) {
        docEntries[fieldId] = mFieldIndex[realFieldId].getDocEntry(docId);
        if (null == docEntries[realFieldId]) {
          throw new RuntimeException("There is no docEntry for docId='" + docId
              + "'" + " fieldId=" + fieldId + " realFieldId=" + realFieldId);
        }
      }
    }
    
    return getStrObjForKNNService(docEntries);
    
  }
  
  public String getStrObjForKNNService(DocEntry[] docEntries) {
    StringBuffer sb = new StringBuffer();
    
    for (int fieldId = 0; fieldId < FeatureExtractor.mFieldNames.length; ++fieldId) 
    if (mUseFiled[fieldId] && docEntries[fieldId] != null) {
      DocEntry oneEntry = docEntries[fieldId];
      
      for (int k = 0; k < oneEntry.mWordIds.length; ++k) {
        if (k > 0) sb.append('\t');
        sb.append(oneEntry.mWordIds[k] + ":" + oneEntry.mQtys[k]);
      }
      sb.append('\n'); // The server works only on Linux so we don't need a platform-independent newline
      if (oneEntry.mWordIdSeq != null) {
        for (int k = 0; k < oneEntry.mWordIdSeq.length; ++k) {
          if (k > 0) sb.append(' ');
          sb.append(oneEntry.mWordIdSeq[k]);
        }
      } else {
        sb.append("@ ");
        sb.append(oneEntry.mDocLen);
      }
      sb.append('\n'); // The server works only on Linux so we don't need a platform-independent newline
    }
    
    return sb.toString();
  }


  InMemForwardIndex mFieldIndex[] = new InMemForwardIndex[FeatureExtractor.mFieldNames.length];
  boolean           mUseFiled[] = new boolean[FeatureExtractor.mFieldNames.length];
}
