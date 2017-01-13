/*
 *  Copyright 2017 Carnegie Mellon University
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

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;

import cz.brmlab.yodaqa.analysis.question.FocusGenerator;
import cz.brmlab.yodaqa.analysis.question.FocusNameProxy;
import cz.brmlab.yodaqa.provider.OpenNlpNamedEntities;
import cz.brmlab.yodaqa.model.Question.Focus;
import info.ephyra.questionanalysis.atype.AnswerType;
import info.ephyra.questionanalysis.atype.FocusFinder;
import info.ephyra.questionanalysis.atype.QuestionClassifier;
import info.ephyra.questionanalysis.atype.QuestionClassifierFactory;
import edu.cmu.lti.javelin.util.Language;
import edu.cmu.lti.oaqa.knn4qa.collection_reader.SQuADIntermCollectionReader;
import edu.cmu.lti.oaqa.knn4qa.types.*;
import edu.cmu.lti.util.Pair;
import de.tudarmstadt.ukp.dkpro.core.languagetool.LanguageToolLemmatizer;
import de.tudarmstadt.ukp.dkpro.core.opennlp.OpenNlpSegmenter;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.StanfordParser;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;

/**
 * This is a question analysis annotator that combines some of the features
 * produced by EphyraQA and YodaQA question analysis modules.
 * 
 * @author Leonid Boytsov
 *
 */
public class QuestionAnalysis extends JCasAnnotator_ImplBase {
  private AnalysisEngine mYodaEngine;
  private JCasFactory    mJCasFactory;
  
  @Override
  public void initialize(UimaContext aContext) throws ResourceInitializationException {
    super.initialize(aContext);
    
    // 1. YodaQA initialization    
    mYodaEngine = createEngine(createYodaQADescription());
    mJCasFactory = new JCasFactory(mYodaEngine);
    // 2. Ephyra initialization
    try {
      mEphyraQClassifier = QuestionClassifierFactory.getInstance(
      new Pair<Language,Language>(Language.valueOf("en_US"),Language.valueOf("en_US")));
    } catch (Exception e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
  }

  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    JCas questView = null;
    try {
      questView = aJCas.getView(SQuADIntermCollectionReader.QUESTION_VIEW);
    } catch (CASException e) {
      throw new AnalysisEngineProcessException(new Exception("No question view in the CAS!"));      
    }

    for (FactoidQuestion q : JCasUtil.select(questView, FactoidQuestion.class)) {
      JCas tmpJCas = null;
      try {
        tmpJCas = mJCasFactory.borrowJCas();
        tmpJCas.setDocumentLanguage(SQuADIntermCollectionReader.DOCUMENT_LANGUAGE);
        tmpJCas.setDocumentText(q.getCoveredText());
    
        HashSet<String> hFoci = new HashSet<String>();
        HashSet<String> hQTypes = new HashSet<String>();
        HashSet<String> hWWords = new HashSet<String>();
  
        
        // 1 Ephyra Analysis focus word/phrase
        String focusEphyra = null;
        { 
          focusEphyra = FocusFinder.findFocusWord(q.getCoveredText());
          
          // Ephyra question types
          for (String t1:  getAtypes(q.getCoveredText())) {
            for (String t2 : t1.split("\\.")) {
              hQTypes.add(t2);
            }
          }          
        }
        
        // 2. YodaQA analysis (extract only foci)
        {
          mYodaEngine.process(tmpJCas);
          for (Focus f : JCasUtil.select(tmpJCas, Focus.class)) {
            String fs = procFocus(f.getToken().getLemma().getValue());
            hFoci.add(fs);
          }
          for (Token t : JCasUtil.select(tmpJCas, Token.class)) {
            String tokLemma = t.getLemma().getValue().toLowerCase();
            if (t.getPos().getPosValue().startsWith("W")) {
              hWWords.add(tokLemma);
            }
            if (focusEphyra != null && t.getCoveredText().compareToIgnoreCase(focusEphyra) == 0) { 
              hFoci.add(procFocus(tokLemma));              
            }
          }
        }
        // 3. Now add annotations obtained from the joint analysis of the question
        for (String focus : hFoci) {
          FocusPhrase fw = new FocusPhrase(questView, q.getBegin(), q.getEnd());
          fw.setValue(focus);
          fw.addToIndexes();
        }
        for (String qtype : hQTypes) {
          FactoidQuestionType qt = new FactoidQuestionType(questView, q.getBegin(), q.getEnd());
          qt.setValue(qtype);
          qt.addToIndexes();
        }
        for (String wword : hWWords) {
          WWord ww = new WWord(questView, q.getBegin(), q.getEnd());
          ww.setValue(wword);
          ww.addToIndexes();
        }
        
      } catch (Exception e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      } finally {
        if (tmpJCas != null) mJCasFactory.returnJCas(tmpJCas);
      }
    }
  }
  
  private String procFocus(String value) {
    return value.replaceAll("\\s", "_").toLowerCase();
  }

  AnalysisEngineDescription createYodaQADescription() throws ResourceInitializationException {
    AggregateBuilder builder = new AggregateBuilder();

    /* Token features: */

    builder.add(AnalysisEngineFactory.createEngineDescription(OpenNlpSegmenter.class));

    /* POS, constituents, dependencies: */
    // fast, reliable
    builder.add(AnalysisEngineFactory.createEngineDescription(StanfordParser.class,
          StanfordParser.PARAM_WRITE_POS, true));

    /* Lemma features: */

    // fastest and handling numbers correctly:
    builder.add(AnalysisEngineFactory.createEngineDescription(LanguageToolLemmatizer.class));

    /* Named Entities: */
    builder.add(OpenNlpNamedEntities.createEngineDescription());

    /* Okay! Now, we can proceed with our key tasks. */

    builder.add(AnalysisEngineFactory.createEngineDescription(FocusGenerator.class));
    builder.add(AnalysisEngineFactory.createEngineDescription(FocusNameProxy.class));
    
    AnalysisEngineDescription aed = builder.createAggregateDescription();
    aed.getAnalysisEngineMetaData().setName("edu.cmu.lti.oaqa.knn4qa.annotators.YodaQuestionAnalysis");
    return aed;    
  }
  
  private String[] getAtypes(String question) {
    List<AnswerType> atypes = new ArrayList<AnswerType>();
    try {
      atypes = mEphyraQClassifier.getAnswerTypes(question);
    } catch (Exception e) {
      e.printStackTrace();
    }
    Set<AnswerType> remove = new HashSet<AnswerType>();
    for (AnswerType atype : atypes) {
      if (atype.getFullType(-1).equals("NONE")) {
        remove.add(atype);
      }
    }
    for (AnswerType atype : remove) {
      atypes.remove(atype);
    }
    String[] res = new String[atypes.size()];
    for (int i = 0; i < atypes.size(); i++) {
      String atype = atypes.get(i).getFullType(-1).toLowerCase().replaceAll("\\.", ".NE").replaceAll("^", "NE");
      StringBuilder sb = new StringBuilder(atype);
      Matcher m = Pattern.compile("_(\\w)").matcher(atype);
      while (m.find()) {
        sb.replace(m.start(), m.end(), m.group(1).toUpperCase());
        m = Pattern.compile("_(\\w)").matcher(sb.toString());
      }
      res[i] = sb.toString();
    }
    return res;
  }
  
  private QuestionClassifier mEphyraQClassifier;
}
