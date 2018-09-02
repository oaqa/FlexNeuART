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

import java.util.Arrays;

import javax.annotation.Nullable;

import edu.cmu.lti.oaqa.knn4qa.utils.StringUtilsLeo;

/**
 * This is a configurable feature extractor designed for experimentation (the
 * description format is given below).
 * 
 * <p>The description format:</p>
 * <ul>
 * <li>Descriptions are plus-separated entities in the form: <feature/option name> = <field-specifiers>
 * <li>In the case of feature names, a field specifier is simply a comma-separated list of fields, for 
 * which we activate the feature. For example, <i>bm25=text,text_unlemm</i> activates BM25 for fields
 * <i>text</i>, and <i>text_unlemm</i>.
 * <li>In the case of options names, a field specifier is a comma-separated list of entities in the format
 * <field-name> : <value>. For example, <i>minProbModel1=text:0.001,text_unlemm=0.002</i> sets
 * a minimum Model1 probability for the field <i>text</i> equal to 0.001 and a minimum Model 1
 * probability for the field <i>text_unlemm</i> equal to 0.002.
 * <li>For the list of properties see {@link #getPropertyFieldArrayByName(String)) method.}</li>
 * </ul>
 * 
 * @author Leonid Boytsov
 *
 */
public class InMemIndexFeatureExtractorExperOld extends InMemIndexFeatureExtractorOld {
  public static final String CODE = "experOld";
  
  public static final String USE_ONLY_WEIGHTED_AVG_EMBED = "onlyWghtAvgEmbed";
 
  @Override
  public float getMinModel1Prob(int fieldId) { return mMinModel1Prob[fieldId]; }
  @Override
  public float getMinSimpleTranProb(int fieldId) { return mMinSimpleTranProb[fieldId]; }
  @Override
  public float getMinJSDCompositeProb(int fieldId) { return mMinJSDCompositeProb[fieldId]; }
  @Override
  public float getModel1Lambda(int fieldId) { return mModel1Lambda[fieldId]; }
  @Override
  public float getProbSelfTran(int fieldId) { return mProbSelfTran[fieldId]; }

  
  public InMemIndexFeatureExtractorExperOld(
      String  params,
      String  dirTranPrefix, 
      int     gizaIterQty,
      String  indexPrefix,
      String  embedDir,
      @Nullable String[]  embedFiles,
      @Nullable String[]  highOrderModelFiles) throws Exception {
    super(dirTranPrefix, gizaIterQty, indexPrefix, embedDir, embedFiles, highOrderModelFiles);
    // We will make two passes. 
    // In the first pass, we will read the list of enabled features
    // In the second pass, we will read minimum probabilities and verify if
    // respective fields were activated.
    for (String opt: params.split("[+]")) {
      if (opt.isEmpty()) continue;
      
      if (opt.compareToIgnoreCase(USE_ONLY_WEIGHTED_AVG_EMBED) == 0) {
        mUseNonWghtAvgEmbed = false;
        continue;
      }
      
      String parts[] = opt.split("=");
      if (parts.length != 2) 
        throw new Exception("Wrong option: '" + opt + "', options should be in the format: <property key>=<comma-separated definitions/field names>");
      String key = parts[0];
      if (null != getPropertyFieldArrayByName(key)) continue;
      boolean [] featFieldArr = getFeatFieldArrayByName(key);
      if (featFieldArr == null)
        throw new Exception("Neither a property nor a feature name: '" + key + "'");
      for (String fieldName : parts[1].split(",")) {
        int iField = StringUtilsLeo.findInArrayNoCase(fieldName, mFieldNames);
        if (iField < 0)
          throw new Exception("Wrong field name: '" + fieldName + "'");
        featFieldArr[iField] = true;
      }     
    }
    
    mMinModel1Prob       = Arrays.copyOf(mMinModel1ProbDefault, mMinModel1ProbDefault.length);
    mMinJSDCompositeProb = Arrays.copyOf(mMinJSDCompositeProbDefault, mMinJSDCompositeProbDefault.length); 
    mMinSimpleTranProb   = Arrays.copyOf(mMinSimpleTranProbDefault, mMinSimpleTranProbDefault.length);
    mModel1Lambda        = Arrays.copyOf(mModel1LambdaDefault, mModel1LambdaDefault.length);
    mProbSelfTran        = Arrays.copyOf(mProbSelfTranDefault, mProbSelfTranDefault.length);

    // Second pass
    for (String opt: params.split("[+]")) {
      if (opt.isEmpty()) continue;
      if (opt.compareToIgnoreCase(USE_ONLY_WEIGHTED_AVG_EMBED) == 0) continue;
      String parts[] = opt.split("=");
      if (parts.length != 2) 
        throw new Exception("Wrong option: '" + opt + "', options should be in the format: <property key>=<comma-separated definitions/field names>");
      String key = parts[0];
      if (getFeatFieldArrayByName(key) != null) continue;
      float probFieldArr[] = getPropertyFieldArrayByName(key);
      if (null == probFieldArr) throw new Exception("Bug: unrecognized option '" + key + "'");
      for (String def: parts[1].split(",")) {
        String parts1[] = def.split(":");
        float  val;
        if (parts1.length != 2) 
          throw new Exception("Wrong definition of a property: '" + def + "', expecting <field name>:<property value>");
        try {
          val = Float.parseFloat(parts1[1]);
        } catch (NumberFormatException e) {
          throw new Exception("In the definition of a property: '" + def + "', the second component is not float!");
        }
        String fieldName = parts1[0];
        int iField = StringUtilsLeo.findInArrayNoCase(fieldName, mFieldNames);
        if (iField < 0)
          throw new Exception("Wrong field name: '" + fieldName + "'");
        probFieldArr[iField] = val;
      }      
    }
  }
  
  int getFieldIdByName(String fieldName) throws Exception {
    int id = StringUtilsLeo.findInArrayNoCase(fieldName, mFieldNames);
    if (id < 0) throw new Exception("Not a field name: " + fieldName);
    return id;
  }
  
  boolean[] getFeatFieldArrayByName(String featName) {
    featName = featName.toLowerCase();
    if (featName.equals("avg_embed"))     return mUseAveragedEmbedFeatures;
    if (featName.equals("avg_embed_bm25"))return mUseAveragedEmbedBM25Features;
    if (featName.equals("wmd"))           return mUseWMDFeatures;
    if (featName.equals("lcs_embed"))     return mUseLCSEmbedFeatures;
    if (featName.equals("bm25"))          return mUseBM25FeatureQueryNorm;
    if (featName.equals("overall_match")) return mUseOverallMatchFeatureQueryNorm;
    if (featName.equals("model1"))        return mUseModel1FeatureQueryNorm;
    if (featName.equals("simple_tran"))   return mUseSimpleTranFeatureQueryNorm;
    if (featName.equals("jsd_composite")) return mUseJSDComposite;
    if (featName.equals("tfidf"))         return mUseTFIDFFeatureQueryNorm;
    if (featName.equals("cosine"))        return mUseCosineText;
    if (featName.equals("lcs"))           return mUseLCSFeatureQueryNorm; 
    return null;
  }
  
  float[] getPropertyFieldArrayByName(String key) {
    if (key.equalsIgnoreCase("minProbModel1"))          return mMinModel1Prob;
    if (key.equalsIgnoreCase("minProbSimpleTran"))      return mMinSimpleTranProb; 
    if (key.equalsIgnoreCase("minProbJSDComposite"))    return mMinJSDCompositeProb;
    if (key.equalsIgnoreCase("model1Lambda"))           return mModel1Lambda;
    if (key.equalsIgnoreCase("probSelfTran"))           return mProbSelfTran;
    return null;
  }

  // By default all features are reset to zero
  private boolean[] mUseAveragedEmbedFeatures         = new boolean[mFieldNames.length];
  private boolean[] mUseAveragedEmbedBM25Features     = new boolean[mFieldNames.length];
  private boolean[] mUseWMDFeatures                   = new boolean[mFieldNames.length];
  private boolean[] mUseLCSEmbedFeatures              = new boolean[mFieldNames.length];
  private boolean[] mUseBM25FeatureQueryNorm          = new boolean[mFieldNames.length];
  private boolean[] mUseOverallMatchFeatureQueryNorm  = new boolean[mFieldNames.length];
  private boolean[] mUseModel1FeatureQueryNorm        = new boolean[mFieldNames.length];
  private boolean[] mUseSimpleTranFeatureQueryNorm    = new boolean[mFieldNames.length];
  private boolean[] mUseJSDComposite                  = new boolean[mFieldNames.length];
  private boolean[] mUseTFIDFFeatureQueryNorm         = new boolean[mFieldNames.length];
  private boolean[] mUseCosineText                    = new boolean[mFieldNames.length];
  private boolean[] mUseLCSFeatureQueryNorm           = new boolean[mFieldNames.length];
  
  private float [] mMinModel1Prob       = new float[mFieldNames.length];
  private float [] mMinJSDCompositeProb = new float[mFieldNames.length];
  private float [] mMinSimpleTranProb   = new float[mFieldNames.length];
  private float [] mModel1Lambda        = new float[mFieldNames.length];
  private float [] mProbSelfTran        = new float[mFieldNames.length];
  
  private boolean mUseNonWghtAvgEmbed = true;
  
  @Override
  public boolean useNonWghtAvgEmbed() { 
    return mUseNonWghtAvgEmbed; 
  }
  
  @Override
  public boolean useAveragedEmbedFeatures(int fieldId) {
    return mUseAveragedEmbedFeatures[fieldId];
  }
  
  @Override
  public boolean useAveragedEmbedBM25Features(int fieldId) {
    return mUseAveragedEmbedBM25Features[fieldId];
  }  
  
  @Override
  public boolean useWMDFeatures(int fieldId) {
    return mUseWMDFeatures[fieldId];
  }

  @Override
  public boolean useLCSEmbedFeatures(int fieldId) {
    return mUseLCSEmbedFeatures[fieldId];
  }

// START of query-normalized features

  @Override
  public boolean useBM25FeatureQueryNorm(int fieldId) {
    return mUseBM25FeatureQueryNorm[fieldId];
  }
  
  @Override
  public boolean useOverallMatchFeatureQueryNorm(int fieldId) {
    return mUseOverallMatchFeatureQueryNorm[fieldId];
  }  

  @Override
  public boolean useModel1FeatureQueryNorm(int fieldId) {
    return mUseModel1FeatureQueryNorm[fieldId];
  }

  @Override
  public boolean useSimpleTranFeatureQueryNorm(int fieldId) {
    return mUseSimpleTranFeatureQueryNorm[fieldId];
  }  
    
  @Override
  public boolean useJSDCompositeFeatures(int fieldId) {
    return mUseJSDComposite[fieldId];
  }
    
  @Override
  public boolean useTFIDFFeatureQueryNorm(int fieldId) {
    return mUseTFIDFFeatureQueryNorm[fieldId];
  }
    
  @Override
  public boolean useLCSFeatureQueryNorm(int fieldId) {
    return mUseLCSFeatureQueryNorm[fieldId];
  }  

// END of query-normalized features
  
  @Override
  public boolean useCosineTextFeature(int fieldId) {
    return mUseCosineText[fieldId];
  }
  
  @Override
  public boolean useBM25Feature(int fieldId) {    
    return false;
  }

  @Override
  public boolean useTFIDFFeature(int fieldId) {    
    return false;
  }
  
  @Override
  public boolean useOverallMatchFeature(int fieldId) {
    return false;
  }

  @Override
  public boolean useModel1Feature(int fieldId) {
    return false;
  }

  @Override
  public boolean useSimpleTranFeature(int fieldId) {
    return false;
  }
  
  
  @Override
  public boolean useLCSFeature(int fieldId) {
    return false;
  }
}
