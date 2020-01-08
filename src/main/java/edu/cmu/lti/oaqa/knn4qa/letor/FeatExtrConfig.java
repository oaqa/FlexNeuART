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
package edu.cmu.lti.oaqa.knn4qa.letor;

import java.io.File;
import java.util.Map.Entry;

import com.google.gson.Gson;

import edu.cmu.lti.oaqa.knn4qa.utils.Const;
import edu.cmu.lti.oaqa.knn4qa.utils.KeyValueConfig;

import org.apache.commons.io.FileUtils;

class OneFeatExtrConf extends KeyValueConfig {
  
	String                  type;
  
  @Override
  public String getName() {
  	return "extractor " + type;
  }
}

/**
 * A wrapper class to read and parse a feature-extractor configuration.
 * 
 * @author Leonid Boytsov
 *
 */
public class FeatExtrConfig {

  
  OneFeatExtrConf[]   extractors;
  
  public static FeatExtrConfig readConfig(String inputFile) throws Exception {
    Gson gson = new Gson();    

    return gson.fromJson(FileUtils.readFileToString(new File(inputFile), Const.ENCODING), 
                         FeatExtrConfig.class);
    
  }
  
  /**
   * Just a stupid testing function.
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    String inputFile = args[0];
   
    FeatExtrConfig tmp = readConfig(inputFile);
    
    for (OneFeatExtrConf extr:tmp.extractors) {
      System.out.println(extr.type);
      System.out.println(extr.getAllParams());
      for(Entry<String, String> e : extr.getAllParams().entrySet())
        System.out.println(e.getKey() + " -> " + e.getValue());
    }
  }  
}


