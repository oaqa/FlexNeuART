package edu.cmu.lti.oaqa.flexneuart.giza;

import java.util.ArrayList;

/**
 * Translation probabilities for one source word.
 * 
 * @author Leonid Boytsov
 *
 */
public class GizaOneWordTranRecs {
  /**
   * Constructor.
   * 
   * @param sortedEntries partial (no source ID) translation entries, should be sorted
   *        in the order of ascending mDstIds.
   */
  public GizaOneWordTranRecs(ArrayList<TranRecNoSrcId> sortedEntries) {
    int n = sortedEntries.size();
    mProbs = new float[n];
    mDstIds = new int[n];
    for (int i = 0; i < n; ++i) {
      mProbs[i] = sortedEntries.get(i).mProb;
      mDstIds[i] = sortedEntries.get(i).mDstId;
    }
  }
  
  final public float [] mProbs;
  final public int   [] mDstIds;
}
