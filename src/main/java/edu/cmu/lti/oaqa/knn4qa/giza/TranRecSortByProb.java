/*
 *  Copyright 2019 Carnegie Mellon University
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
package edu.cmu.lti.oaqa.knn4qa.giza;

public class TranRecSortByProb implements Comparable<TranRecSortByProb> {
  public TranRecSortByProb(int mDstWorId, float mProb) {
    this.mDstWorId = mDstWorId;
    this.mProb = mProb;
  }
  final public int     mDstWorId;
  final public float   mProb;
  @Override
  public int compareTo(TranRecSortByProb o) {
    // If mProb > o.mProb, we return -1
    // that is higher-probability entries will go first
    return (int) Math.signum(o.mProb - mProb);
  }
}