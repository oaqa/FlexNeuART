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

public class CandidateEntry implements Comparable<CandidateEntry>, java.io.Serializable {
  private static final long serialVersionUID = 1L;
  public final String mDocId;
  public final float mOrigScore;
  public float mScore;
  
  public int         mRelevGrade = 0;
  public int         mOrigRank = 0;


  public CandidateEntry(String docId, float score) {
    mDocId = docId;
    mOrigScore = mScore = score;
  }

  @Override
  public int compareTo(CandidateEntry o) {
    // If mScore is greater => result is -1
    // That is the greater entry is ranked earlier
    return (int) Math.signum(o.mScore - mScore);
  }
  
  /**
   * Create a list of candidate entries with zero scores from an array of document IDs.
   * 
   * @param docIds
   * @return
   */
  public static CandidateEntry[] createZeroScoreCandListFromDocIds(ArrayList<String> docIds) {
    CandidateEntry[] res = new CandidateEntry[docIds.size()];
    
    for (int i = 0; i < docIds.size(); i++) {
      res[i] = new CandidateEntry(docIds.get(i), 0);
    }
    
    return res;
  }
};
