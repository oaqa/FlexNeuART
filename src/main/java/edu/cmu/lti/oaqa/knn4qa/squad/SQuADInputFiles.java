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
package edu.cmu.lti.oaqa.knn4qa.squad;


public class SQuADInputFiles {
  /**
   * Original dev file name, as well as file names of the split training part.
   */
  public final static String[] mInputFiles = {
    "dev-v1.1.json.gz",
    "train-v1.1-split_dev1.gz",
    "train-v1.1-split_dev2.gz",
    "train-v1.1-split_train.gz",
    "train-v1.1-split_tran.gz"
  };
  /**
   * File names for input files in the intermediate JSON format (classes QAData, QAPassage, QAQuestion).
   */
  public final static String[] mIntermInputFiles = {
    "squad-interm_test.gz",
    "squad-interm_dev1.gz",
    "squad-interm_dev2.gz",
    "squad-interm_train.gz",
    "squad-interm_tran.gz"    
  };
  
  static {
    if (mIntermInputFiles.length != mInputFiles.length) {
      System.err.println("Fatal bug: mIntermInputFiles.length != mInputFiles.length");
      System.exit(1);
    }
  };
  
}
