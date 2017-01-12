package edu.cmu.lti.oaqa.knn4qa.annotators;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.fit.util.JCasUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import edu.cmu.lti.oaqa.knn4qa.qaintermform.*;
import edu.cmu.lti.oaqa.knn4qa.types.*;
import edu.cmu.lti.oaqa.knn4qa.utils.CompressUtils;

import java.io.*;

/**
 * 
 * This class creates annotations by reading annotations created previously
 * by a Spacy pipeline (script: scripts/data/run_spacy_ner.py).
 * 
 * @author Leonid Boytsov
 *
 */
public class SpacyNERReaderAnnot extends JCasAnnotator_ImplBase {
  private static final Logger logger = LoggerFactory.getLogger(SpacyNERReaderAnnot.class);
  
  private static final String PARAM_SPACY_NER_FILE = "SpacyNERFile";
  
  public static final String SPACY_NER_TYPE = "spacy_ner";
  
  @ConfigurationParameter(name = PARAM_SPACY_NER_FILE, mandatory = true)
  private String mSpacyNerFileName;
  
  private BufferedReader mSpacyNer;
  
  private static Gson    mGSON = new Gson();
  
  @Override
  public void initialize(UimaContext aContext)
  throws ResourceInitializationException {
    super.initialize(aContext);
    
    try {
      mSpacyNer = 
          new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(mSpacyNerFileName)));
    } catch (IOException e) {
      logger.error("Error opening file: " + mSpacyNerFileName);
      throw new ResourceInitializationException(e);
    }
  }
  
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    Passage passage = null;
    for (Passage p : JCasUtil.select(aJCas, Passage.class)) { passage = p ; break; } // we have only one passage
    // Now retrieve all annotation entries until we find one with the id the has the same id as the passage
    
    NERData passageAnnot = null;
    String passId = passage.getId();
    do {
      String line;
      try {
        line = mSpacyNer.readLine();
      } catch (IOException e) {
        e.printStackTrace();
        throw new AnalysisEngineProcessException(e);
      }
      if (line == null) {
        throw new AnalysisEngineProcessException(
            new Exception("Reached EOF in '" + mSpacyNerFileName + "' before finding record for passage id='" + passId)); 
      }
      try {
        passageAnnot = mGSON.fromJson(line, NERData.class);
      } catch (JsonSyntaxException e) {
        logger.error("Cannot parse NER for line:" + line);
        throw new AnalysisEngineProcessException(e);
      }
    } while (!passageAnnot.id.equals(passId));
    /*
     * Here we end up with the annotation record that has the same ID as the passage.
     * If we reach EOF before finding such an annotation record, an exception is thrown.
     * Note that the order of annotations is the same as the order of CASes.
     * So, if there is always an annotation record for each passage, it will be found.
     */
    if (passageAnnot.annotations != null) {
      for (NERAnnotation an : passageAnnot.annotations) {
        Entity e = new Entity(aJCas, an.start, an.end);
        e.setEtype(SPACY_NER_TYPE);
        e.setLabel(an.label);
        e.addToIndexes();        
      }
    }

  }

}
