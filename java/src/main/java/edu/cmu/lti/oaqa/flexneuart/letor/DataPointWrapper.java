/*
 *  Copyright 2014+ Carnegie Mellon University
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
package edu.cmu.lti.oaqa.flexneuart.letor;

import ciir.umass.edu.learning.DataPoint;
import no.uib.cipr.matrix.DenseVector;

/**
 * This class converts a dense vector to DataPoint format of the RankLib library
 * version 2.5.
 * 
 * <p>Perhaps, an <b>important</b> note: a RankLib DataPoint class contains 
 * a static variable featureCount. It doesn't seem to be used except
 * for feature normalization or by the evaluator code. So, it seems
 * to be fine to leave this variable set to default value (zero).

 * </p>
 * 
 * @author Leonid Boytsov
 *
 */
public class DataPointWrapper extends DataPoint {

  public void assign(DenseVector feat) {
    mFeatValues = new float[feat.size() + 1];
    double data[] = feat.getData();
    for (int i = 0; i < feat.size(); ++i)
      mFeatValues[i+1] = (float)data[i];
  }
  
  @Override
  public float getFeatureValue(int fid) {
    return mFeatValues[fid];
  }

  @Override
  public float[] getFeatureVector() {
    return mFeatValues;
  }

  @Override
  public void setFeatureValue(int fid, float val) {
    mFeatValues[fid] = val;
  }

  @Override
  public void setFeatureVector(float[] vals) {
    mFeatValues = vals;
  }
  
  float [] mFeatValues;
}
