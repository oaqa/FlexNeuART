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
import java.util.HashMap;

/**
 * 
 * This class creates annotations by reading annotations created previously
 * by a Spacy pipeline (script: scripts/data/run_spacy_ner.py).
 * 
 * <p>Important note: it will load the text of annotations into memory 
 * in the form of a hash map (on long String per document). So for Wikipedia, this should
 * be run on a machine with about 32GB+ memory.</p>
 * 
 * <p>However, this annotator can be deployed in different threads, because it will
 * keep only one copy of the annotation hash map.</p>
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
  
  private Gson    mGSON = new Gson();
  private static HashMap<String,String>  mNERStore = null; 
  
  @Override
  public void initialize(UimaContext aContext)
  throws ResourceInitializationException {
    super.initialize(aContext);
    
    synchronized (SpacyNERReaderAnnot.class) {
      if (null == mNERStore) {
        // We will read NER file, but only once
        mNERStore = new HashMap<String,String>();
      
        BufferedReader spacyNer = null;
        
        try {
          spacyNer = 
              new BufferedReader(new InputStreamReader(CompressUtils.createInputStream(mSpacyNerFileName)));
          
          logger.info("Starting reading annotations from '" + mSpacyNerFileName + "'");
          
          String line;
                  
          while ((line = spacyNer.readLine()) != null) {
            NERData passageAnnot = mGSON.fromJson(line, NERData.class);
            // Keeping it in the form of a String rather than
            // in a parsed form should greatly reduce memory consumption
            mNERStore.put(passageAnnot.id, line);
          }
          
          logger.info("Read " + mNERStore.size() + " annotations from '" + mSpacyNerFileName + "'");
  
        } catch (Exception e) {
          logger.error("Error opening file: " + mSpacyNerFileName);
          throw new ResourceInitializationException(e);
        }
        System.gc();
      }
    }
    
  }
  
  
  @Override
  public void process(JCas aJCas) throws AnalysisEngineProcessException {
    Passage passage = JCasUtil.selectSingle(aJCas, Passage.class); // Will throw an exception of Passage is missing
    // Now retrieve all annotation entries until we find one with the id the has the same id as the passage
    
    String passId = passage.getId();

    // This doesn't have to be synced, b/c we don't update this hash after 
    // the first call to initialize() is completed.
    String tmp = mNERStore.get(passId);
    // Parsing the same JSON twice is the price we pay to reduce memory footprint of the hash
    NERData passageAnnot = mGSON.fromJson(tmp, NERData.class);
    if (passageAnnot != null) {
      if (passageAnnot.annotations != null) {
        for (NERAnnotation an : passageAnnot.annotations) {
          Entity e = new Entity(aJCas, an.start, an.end);
          e.setEtype(SPACY_NER_TYPE);
          e.setLabel(an.label);
          e.addToIndexes();        
        }
      }
    } else {
      throw new AnalysisEngineProcessException(
          new Exception("Can't retrieve anntotation for the passage id='" + passId + "'"));
    }
  }

}
