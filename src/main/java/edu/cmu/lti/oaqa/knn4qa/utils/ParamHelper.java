package edu.cmu.lti.oaqa.knn4qa.utils;

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
