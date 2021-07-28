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
package edu.cmu.lti.oaqa.flexneuart.cand_providers;

import java.util.ArrayList;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.CommonParams;
import edu.cmu.lti.oaqa.flexneuart.resources.ResourceManager;
import edu.cmu.lti.oaqa.flexneuart.resources.RestrictedJsonConfig;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

/**
 * An ID-mapping provider does not search on its own. It uses an actual
 * provider to carry out a search on (potentially) a very different data representation.
 * For example, to retrieve candidate passages, we may search for complete documents.
 * The job of the ID-mapping provider is to carry out a document search and to subsequently
 * retrieve the list of all passage IDs for the retrieved documents.
 * This requires a special ID-mapping index, which for every document ID stores
 * a list of white-space separated IDs of passages that this document contains.
 * 
 * @author Leonid Boytsov
 *
 */
public class IdMapperCandidateProvider extends CandidateProvider {
  final Logger logger = LoggerFactory.getLogger(IdMapperCandidateProvider.class);
  
  private String mIdMapFieldName;
  private ForwardIndex mIdMapper;
  
  public IdMapperCandidateProvider(ResourceManager resourceManager, 
                                   String provURI,  
                                   RestrictedJsonConfig addConf) throws Exception {
    String provType = addConf.getReqParamStr("candProv");
    String configName = addConf.getParam("candProvAddConf", (String)null);
    if (provType.equalsIgnoreCase(CandidateProvider.CAND_TYPE_ID_MAPPER)) {
      throw new Exception("Recursive use of the ID-mapping provider: " + CandidateProvider.CAND_TYPE_ID_MAPPER);
    }
    mBackendProv = resourceManager.createCandProviders(provType, provURI, configName, 1)[0];
    mIsThreadSafe = mBackendProv.isThreadSafe();
    
    mIdMapFieldName = addConf.getParam(CommonParams.ID_MAP_FIELD_NAME, "");

    if (mIdMapFieldName.length() == 0) {
      mIdMapper = null;
    } else {
      logger.info("Using field: " + mIdMapFieldName + " to map entry/document IDs");
      mIdMapper = resourceManager.getFwdIndex(mIdMapFieldName);
      if (!mIdMapper.isTextRaw()) {
        throw new Exception("Remapping field " + mIdMapFieldName + " should have the raw text type!");
      }
    }
  }

  @Override
  public boolean isThreadSafe() {
    // Note that it depends on the underlying provider
    return mIsThreadSafe;
  }

  @Override
  public String getName() {
    return this.getClass().getName();
  }

  @Override
  public CandidateInfo getCandidates(int queryNum, DataEntryFields queryFields, int maxQty) throws Exception {
    CandidateInfo tmpInfo = mBackendProv.getCandidates(queryNum, queryFields, maxQty);
    
    ArrayList<CandidateEntry>  mappedEntries = new ArrayList<CandidateEntry>();
    for (CandidateEntry e : tmpInfo.mEntries) {
      // The re-mapping index contains white-space separated IDs
      String mappedIdStr = mIdMapper.getDocEntryTextRaw(e.mDocId);
      if (mappedIdStr == null) {
        throw new Exception("Cannot map id '" + e.mDocId + "' using the field: " + mIdMapFieldName);
      }
      for (String mappedId : StringUtils.splitOnWhiteSpace(mappedIdStr)) {
        mappedEntries.add(new CandidateEntry(mappedId, e.mScore, e.mOrigScore));
      }
    }
    CandidateEntry tmpEntries[] = mappedEntries.toArray(new CandidateEntry[0]);
    Arrays.sort(tmpEntries);
    
    int qty = Math.min(maxQty, tmpEntries.length);
    
    return new CandidateInfo(tmpEntries.length, Arrays.copyOf(tmpEntries, qty));
  }

  private final CandidateProvider mBackendProv;
  private final boolean           mIsThreadSafe;
}
