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
package edu.cmu.lti.oaqa.flexneuart.utils;

import java.util.Comparator;

/**
 * A comparator for the id-value pair class, 
 * which sorts by the value (in the descending order)
 * rather than by the ID.
 * 
 * @author Leonid Boytsov
 */
public class IdValParamByValDesc implements Comparator<IdValPair> {

  @Override
  public int compare(IdValPair e1, IdValPair e2) {
    // If e1.mVal > e2.mVal, we return -1
    // that is larger-value entries will go first
    return (int) Math.signum(e2.mVal - e1.mVal);
  }

}
