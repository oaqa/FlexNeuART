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
package edu.cmu.lti.oaqa.knn4qa.annotators;

import java.util.Collection;

import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.slf4j.Logger;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import edu.cmu.lti.oaqa.knn4qa.types.Answer;
import edu.cmu.lti.oaqa.knn4qa.types.Question;

public class PrintInfoHelper {
  public static synchronized void printInfo1(Logger logger, JCas jcas) {
    long threadId = Thread.currentThread().getId();
    String type = "";
    String id = "";
    Collection<Question> q = JCasUtil.select(jcas, Question.class);
    if (!q.isEmpty()) {
      for (Question y: q) {
        id = y.getUri();
        type="question";
        break;
      }
    } else {
      for (Answer an : JCasUtil.select(jcas, Answer.class)) {
        id = an.getId();
        type = "answer";
        break;
      }
    }
    
    Collection<Sentence> s = JCasUtil.select(jcas, Sentence.class);
    int sentQty = s.size();
    
    Collection<Token> t = JCasUtil.select(jcas, Token.class);
    int tokQty = t.size();
    
    logger.info(String.format("Thread # %d Obj Type %s Id %s Sent qty %d token qty %d",
                              threadId,type,id, sentQty, tokQty));
  }
  
}
