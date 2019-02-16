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
package edu.cmu.lti.oaqa.knn4qa.simil;

/**
 * An interface for the distance (not necessarily metric distance).
 * The closer are the objects, the smaller is the distance. The minimum
 * value of the distance should be zero.
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class AbstractDistance {
  public static final String L2 = "l2";
  private static final String EUCLIDEAN = "euclidean";
  public static final String COSINE = "cosine";

  public abstract float compute(float [] vec1, float [] vec2);
  
  /**
   * @return  human-readable distance name.
   */
  public abstract String getName();
  
  public static AbstractDistance create(String name) throws Exception {
    name = name.toLowerCase().trim();
    if (name.equals(L2) || name.equals(EUCLIDEAN))
      return new EuclideanDistance();
    if (name.equals(COSINE))
      return new CosineDistance();
    
    throw new Exception("Unknown distance: " + name);
  }
}
