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
package edu.cmu.lti.oaqa.knn4qa.collection_reader;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * A class that represents a single parsed answer.
 * 
 * @author Leonid Boytsov
 *
 */
/**
 * @author leo
 *
 */
public class ParsedQuestion {
  private static final Logger logger = LoggerFactory.getLogger(ParsedQuestion.class);
  
  public final String                mQuestion;
  public final String                mQuestDetail;
  public final String                mQuestUri;
  public final int                   mBestAnswId;
  public final ArrayList<String>     mAnswers;
  
  
  /**
   * Constructor, attention it modifies the parameter answer (to keep members final).
   * 
   * @param quest           main question
   * @param questDetail     question details
   * @param questUri        question URI
   * @param answers         an array of answers
   * @param bestAnsw        best answer
   * @param doCleanUp       do we do the clean up?
   */
  ParsedQuestion(String quest, String questDetail, String questUri,
                 ArrayList<String> answers, String bestAnsw, 
                 boolean doCleanUp) {
    mQuestion    = cleanUpWrapper(doCleanUp, quest); 
    mQuestDetail = cleanUpWrapper(doCleanUp, questDetail); 
    mQuestUri    = questUri;
    bestAnsw     = cleanUpWrapper(doCleanUp, bestAnsw);
    
    int bestAnswId = -1;
    for (int i = 0; i < answers.size(); ++i) {
      answers.set(i, cleanUpWrapper(doCleanUp, answers.get(i)));
      if (answers.get(i).equals(bestAnsw)) {
        bestAnswId = i;
// No break here: we need to finish the clean-up!        
//        break;
      }
    }

    if (bestAnswId < 0) {
      logger.info("The best answer is missing among answers for uri='"
          + mQuestUri + "'");
      answers.add(0, bestAnsw);
      bestAnswId = 0;
    }
    
    mAnswers     = answers;
    mBestAnswId = bestAnswId;   
    
  }
  
  private static String cleanUpWrapper(boolean doCleanUp, String s) {
    return doCleanUp ? cleanUp(s) : s;
  }
  
  private static Pattern mReplBR = Pattern.compile("<br\\s*/?>",
      Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);
  private static Pattern mReplTags = Pattern.compile("<[a-z]+[^/>]*/?>",
      Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);

  /**
   * Removing accents. Taken from
   * http://www.drillio.com/en/software-development/
   * java/removing-accents-diacritics-in-any-language/
   */
  public static String removeDiacritics(String text) {
    return text == null ? null : Normalizer.normalize(text, Form.NFD)
        .replaceAll("\\p{InCombiningDiacriticalMarks}+", "");
  }

  public static String cleanUp(String s) {
    s = s.trim();
    s = s.replaceAll("\r+", ""); // "\r" may come from a file in DOS encoding;
    s = s.replace('’', '\''); // ugly hack for Yahoo answers

    s = removeDiacritics(s);
    s = s.replaceAll("[^\\x00-\\x7F]", " "); // remove non-ASCII

    /*
     * Repeating punctuation marks cause all kind of trouble in ClearNLP
     * including infinite loops and stack overflow.
     */
    s = s.replaceAll("[?]+", "?");
    s = s.replaceAll("[!]+", "!");
    s = s.replaceAll("[.]+", ".");
    s = s.replaceAll("[:]+", ":");

    Matcher m1 = mReplBR.matcher(s);
    s = m1.replaceAll("\n");
    Matcher m2 = mReplTags.matcher(s);
    return m2.replaceAll(" ").replaceAll("\n+", "\n");
  }  
  
  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "ParsedQuestion [mQuestion=" + mQuestion + ", mQuestDetail="
        + mQuestDetail + ", mQuestUri=" + mQuestUri + ", mBestAnswId="
        + mBestAnswId + ", mAnswers=" + mAnswers + "]";
  }
  
  private void reportDiffString(String s1, String s2, String Name) {
    if (s1 == null && s2 != null) {
      logger.error(Name + ": the first is null, but the second is not");
    }
    if (s1 != null && s2 == null) {
      logger.error(Name + ": the first is not null, but the second is");
    }    
    if (s1 != null && s2 != null && !s1.equals(s2)) {
      logger.error(Name + ": different");
    }        
  }

  public boolean compare(Object obj, boolean bLogDiff) {
    if (this == obj)
      return true;
    if (obj == null) {
      logger.error("The second object is null");
      return false;
    }
    if (getClass() != obj.getClass()) {
      logger.error("Different object classes");
      return false;
    }
    ParsedQuestion other = (ParsedQuestion) obj;
    
    if (mAnswers == null) {
      if (other.mAnswers != null) {
        logger.error("One answer array is null, but the other is not");
        return false;
      }
    } else if (!mAnswers.equals(other.mAnswers)) {
      logger.error("Answers are different");
      if (mAnswers.size() != other.mAnswers.size()) {
        logger.error("Answer sizes are different");
      }
      else {
        for (int i = 0; i < mAnswers.size(); ++i) {
          if (!mAnswers.get(i).equals(other.mAnswers.get(i))) {
            logger.error("Answers id= " + i + " are different");
          }
        }
      }
      return false;
    }
    if (mBestAnswId != other.mBestAnswId) {
      logger.error("Best answer IDs are different");
      return false;
    }
    
    reportDiffString(mQuestDetail, other.mQuestDetail, "mQuestDetail");
    reportDiffString(mQuestion, other.mQuestion, "mQuestion");
    reportDiffString(mQuestUri, other.mQuestUri, "mQuestUri");
    
    if (mQuestDetail == null) {
      if (other.mQuestDetail != null) {
        return false;
      }
    } else if (!mQuestDetail.equals(other.mQuestDetail)) {
      return false;
    }
    if (mQuestUri == null) {
      if (other.mQuestUri != null) {
        return false;
      }
    } else if (!mQuestUri.equals(other.mQuestUri)) {
      return false;
    }
    if (mQuestion == null) {
      if (other.mQuestion != null)
        return false;
    } else if (!mQuestion.equals(other.mQuestion)) {
      return false;
    }
    
    return true;
  } 
}
