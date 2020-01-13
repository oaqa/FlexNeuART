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
package edu.cmu.lti.oaqa.knn4qa;

import org.slf4j.Logger;

public class AbstractTest {

  public AbstractTest() {
    super();
  }

  protected boolean approxEqual(Logger logger, double f1, double f2, double threshold) {
    boolean res = Math.abs(f1 - f2) < threshold;
    logger.info(String.format("Comparing %f vs %f with threshold %f, result %b",
                              f1, f2, threshold, res));
    return res;
  }

}
