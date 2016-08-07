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

package edu.cmu.lti.oaqa.knn4qa.letor;

/**
 * 
 * All good features available.
 * 
 * @author Leonid Boytsov
 *
 */
public class InMemIndexFeatureExtractorComplex extends InMemIndexFeatureExtractor {
  public static final String CODE = "complex";

  public InMemIndexFeatureExtractorComplex(String dirTranPrefix, int gizaIterQty,
      String indexPrefix, String embedDir, String[] embedFiles, String[]  highOrderModelFiles) throws Exception {
    super(dirTranPrefix, gizaIterQty, indexPrefix, embedDir, embedFiles, highOrderModelFiles);
  }

  
  private boolean isEffectiveField(int fieldId) {
    return fieldId == FeatureExtractor.TEXT_FIELD_ID ||
           fieldId == FeatureExtractor.TEXT_UNLEMM_FIELD_ID ||
           fieldId == FeatureExtractor.BIGRAM_FIELD_ID;
  }
  
// BASIC FEATURES
  
  @Override
  public boolean useBM25Feature(int fieldId) {    
    return isEffectiveField(fieldId); 
  }

  @Override
  public boolean useBM25FeatureQueryNorm(int fieldId) {    
    return isEffectiveField(fieldId);
  }  
  
  
  @Override
  public boolean useOverallMatchFeature(int fieldId) {
    return isEffectiveField(fieldId);
  }
  
  @Override
  public boolean useOverallMatchFeatureQueryNorm(int fieldId) {
    return isEffectiveField(fieldId);
  }
  
  @Override
  public boolean useTFIDFFeature(int fieldId) {    
    return isEffectiveField(fieldId);
  }
  
  @Override
  public boolean useTFIDFFeatureQueryNorm(int fieldId) {    
    return isEffectiveField(fieldId);
  }
  
  // END BASIC FEATURES

  // TRANSLATION FEATURES
  
  @Override
  public boolean useModel1Feature(int fieldId) {    
    return fieldId == FeatureExtractor.TEXT_FIELD_ID;
  }
  
  @Override
  public boolean useModel1FeatureQueryNorm(int fieldId) {    
    return fieldId == FeatureExtractor.TEXT_FIELD_ID;
  }  

  @Override
  public boolean useSimpleTranFeature(int fieldId) {
    return fieldId == FeatureExtractor.TEXT_FIELD_ID;
  }
  
  @Override
  public boolean useSimpleTranFeatureQueryNorm(int fieldId) {
    return fieldId == FeatureExtractor.TEXT_FIELD_ID;
  }  

  // END OF TRANSLATION FEATURES
  
  @Override
  public boolean useJSDCompositeFeatures(int fieldId) {
    return fieldId == FeatureExtractor.TEXT_UNLEMM_FIELD_ID;
  }  
  
  @Override
  public boolean useWMDFeatures(int fieldId) {    
    return fieldId == FeatureExtractor.TEXT_UNLEMM_FIELD_ID;
  }

  @Override
  public boolean useLCSEmbedFeatures(int fieldId) {    
    return fieldId == FeatureExtractor.TEXT_UNLEMM_FIELD_ID;
  }

  @Override
  public boolean useAveragedEmbedFeatures(int fieldId) {
    return fieldId == FeatureExtractor.TEXT_UNLEMM_FIELD_ID;
  }
      

  // DISABLED FEATURES

  @Override
  public boolean useLCSFeature(int fieldId) {
    return false;
  }
  
  @Override
  public boolean useLCSFeatureQueryNorm(int fieldId) {
    return false;
  }    

  @Override
  public boolean addRankScores() {
    return true;
  }  
  
}
