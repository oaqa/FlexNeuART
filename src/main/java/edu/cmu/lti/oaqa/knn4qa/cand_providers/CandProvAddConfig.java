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
package edu.cmu.lti.oaqa.knn4qa.cand_providers;

import java.io.File;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import com.google.gson.Gson;

import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.KeyValueConfig;


/**
 * A wrapper class to read and parse additional options for the
 * candidate provider config. Main options (type and index location)
 * should be parsed via apps command line. 
 * 
 * @author Leonid Boytsov
 *
 */
public class CandProvAddConfig extends KeyValueConfig {
	
	protected String mProvName;

	public CandProvAddConfig(Map<String, String> conf, String provName) {
      mProvName = provName;
      this.params = conf;
    }

    @Override
	public String getName() {
		return "candiate provider " + mProvName;
	}
	
	/**
	 * We decided to keep the main parameters outside of this config. However,
	 * it still needs to know its name. For this reason, the reading function
	 * accepts provider name/type as a parameter.
	 * 
	 * @param inputFile		An input JSON file.
	 * @param provName    A provider name.
	 * @return
	 * @throws Exception
	 */
  public static CandProvAddConfig readConfig(String inputFile, String provName) throws Exception {
    Gson gson = new Gson();    

    Map<String, String> conf = gson.fromJson(FileUtils.readFileToString(new File(inputFile), Const.ENCODING), Map.class);
    return new CandProvAddConfig(conf, provName);
  }
  
  /**
   * Just a stupid testing function.
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    String inputFile = args[0];
   
    CandProvAddConfig tmp = readConfig(inputFile, CandidateProvider.CAND_TYPE_LUCENE);
    System.out.println(tmp.getAllParams());
    
  }
  

}
