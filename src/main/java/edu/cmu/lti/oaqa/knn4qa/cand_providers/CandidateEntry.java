package edu.cmu.lti.oaqa.knn4qa.cand_providers;

public class CandidateEntry implements Comparable<CandidateEntry>, java.io.Serializable {
  private static final long serialVersionUID = 1L;
  public final String mDocId;
  public final float mOrigScore;
  public float mScore;
  
  public boolean     mIsRelev = false;
  public int         mOrigRank = 0;


  public CandidateEntry(String docId, float score) {
    mDocId = docId;
    mOrigScore = mScore = score;
  }

  @Override
  public int compareTo(CandidateEntry o) {
    // If mScore is greater => result is -1
    return (int) Math.signum(o.mScore - mScore);
  }
};
