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
package edu.cmu.lti.oaqa.flexneuart.simil_func;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.simil_func.DistanceFunctions;

public class TermMatchSimilarity implements QueryDocSimilarityFunc {
  /**
   * Computes the (normalized) number of query terms that also occur in documents.
   * 
   * @param query
   * @param document
   * 
   * @return the similarity score
   */
  @Override
  public float compute(DocEntryParsed query, DocEntryParsed doc) {
    float score = DistanceFunctions.compOverallMatch(query, doc);

    return score / termQty(query);
  }

  protected int termQty(DocEntryParsed e) {
    int qty = 0;
    for (int i = 0; i < e.mQtys.length; ++i) {
      if (e.mWordIds[i] >= 0) {
        qty += e.mQtys[i];
      }
    }
    if (qty <= 0) {
      qty = 1;
    }
    return qty;
  }

}
