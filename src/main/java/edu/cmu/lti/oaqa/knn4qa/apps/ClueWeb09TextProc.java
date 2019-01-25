package edu.cmu.lti.oaqa.knn4qa.apps;

import java.io.File;
import java.util.ArrayList;

import org.lemurproject.kstem.KrovetzStemmer;

import edu.cmu.lti.oaqa.knn4qa.utils.DictNoComments;

public class ClueWeb09TextProc {

  private KrovetzStemmer mStemmer = new KrovetzStemmer();
  private DictNoComments mStopWords = null;
  private DictNoComments mCommonWords = null;
  private boolean mLowerCase = true;

  private static final String SPACE_REGEXP_SEP = "\\s+";
  
  ClueWeb09TextProc(String stopWordFile, String commonWordFile, boolean lowerCase) throws Exception {
    mLowerCase = lowerCase;
    mStopWords = new DictNoComments(new File(stopWordFile), mLowerCase);
    mCommonWords = new DictNoComments(new File(commonWordFile), mLowerCase);
    
    System.out.println("# of common words to use:" + mCommonWords.getQty());
    System.out.println("# of stop words to use:" + mStopWords.getQty());
  }
  
  public static String[] splitText(String text) {
    return text.split(SPACE_REGEXP_SEP);
  }
  
  /**
   * Remove stop words and keep only words from a list.
   * If the LOWERCASE is true, then all words are lower-cased.
   * 
   * @param text
   * @return
   */
  String filterText(String text) {
    ArrayList<String> res = new ArrayList<String>();
    
    for (String tok : splitText(text)) {
      if (mLowerCase) tok = tok.toLowerCase();
      
      if (!mStopWords.contains(tok) && mCommonWords.contains(tok)) {
        res.add(tok);
      }
    }
    return String.join(" ", res);
  }
  

  String stemText(String text) {
    String toks[] = text.split(SPACE_REGEXP_SEP);
 
    for (int i = 0; i < toks.length; ++i) {
      String s = mStemmer.stem(toks[i]);
      toks[i] = s;
    }
    return String.join(" ", toks);
  }
}
