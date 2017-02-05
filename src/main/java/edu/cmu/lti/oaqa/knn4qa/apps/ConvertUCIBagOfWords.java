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
package edu.cmu.lti.oaqa.knn4qa.apps;

/**
 * An application that can convert a sparse bag-of-words UCI data set to
 * a knn4qa forward file. It accepts two files: docword.<data set name>.txt.gz
 * and vocab.<data set name>.txt  
 * 
 * @author Leonid Boytsov
 *
 */
public class ConvertUCIBagOfWords {

  private static final String INPUT_DIR_PARAM = "input_dir";
  private static final String INPUT_DIR_DESC = "Directory containing input data set";
  private static final String DATASET_NAME_PARAM = "dataset_name";
  private static final String DATASET_NAME_DESC = "Data set name";
  private static final String OUTPUT_FILE_PARAM = "out_file";
  private static final String OUTPUT_FILE_DESC = "Output file";

  public static void main(String[] args) {
    String optKeys[] = {
        INPUT_DIR_PARAM,
        DATASET_NAME_PARAM,
        OUTPUT_FILE_PARAM
    };
    String optDescs[] = {
        INPUT_DIR_DESC,
        DATASET_NAME_DESC,
        OUTPUT_FILE_DESC
    };
    
    boolean hasArg[] = {
        true,
        true,
        true,
    };
    
  }

}
