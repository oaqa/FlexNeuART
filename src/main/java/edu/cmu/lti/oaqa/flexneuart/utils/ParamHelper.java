/*
 *  Copyright 2015+ Carnegie Mellon University
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

import org.apache.commons.cli.*;

/**
 * A command line application parameter helper.
 * 
 * @author Leonid Boytsov
 *
 */
public class ParamHelper {
  /**
   * Constructor: add parameters & parse.
   * 
   * @param args        argument array.
   * @param optKeys     option keys.
   * @param optDescs    option descriptions.
   * @param hasParams   does the option require a parameter?
   */
  public ParamHelper(String args[],
                     String [] optKeys, String [] optDescs, boolean [] hasParams) throws Exception {
    if (optKeys.length != optDescs.length) {
      throw new Exception("The number of keys is not equal to the number of descriptions.");
    }
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < optKeys.length; ++i) {
      mOptions.addOption(optKeys[i], null, hasParams[i], optDescs[i]);
      sb.append(" -" + optKeys[i] + " <" + optDescs[i] + (hasParams[i] ? "":" flag without args") + ">" );
    }
    mParamDesc = sb.toString();
    mCmdLine = mParser.parse(mOptions, args);
  }
  
  public String getParamDesc() {
    return mParamDesc;
  }
  
  public CommandLine getCommandLine() {
    return mCmdLine;
  }
  
  public Options getOptions() { 
    return mOptions; 
  }

  private String            mParamDesc = "";
  private CommandLineParser mParser = new org.apache.commons.cli.GnuParser();
  private Options           mOptions = new Options();
  private CommandLine       mCmdLine = null;
}
