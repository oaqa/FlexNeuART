package edu.cmu.lti.oaqa.flexneuart.apps;

import org.apache.commons.cli.Options;

public class ExportCEDRParams {
  
  public static final String TEST_RUN_FILE_PARAM = "test_run_file";
  public static final String TEST_RUN_FILE_DESC = "a TREC style test/validation run file";
  
  public static final String DATA_FILE_DOCS_PARAM = "data_file_docs";
  public static final String DATA_FILE_DOCS_DESC = "CEDR data file for docs";

  public static final String DATA_FILE_QUERIES_PARAM = "data_file_queries";
  public static final String DATA_FILE_QUERIES_DESC = "CEDR data file for queries";
  
  public static final String QUERY_DOC_PAIR_FILE_PARAM = "train_pairs_file";
  public static final String QUERY_DOC_PAIR_FILE_DESC = "query-document pairs for training";
  
  protected static void addOptionsDesc(Options opts) {
    opts.addOption(ExportCEDRParams.TEST_RUN_FILE_PARAM, null, true, TEST_RUN_FILE_DESC); 
    opts.addOption(ExportCEDRParams.DATA_FILE_DOCS_PARAM, null, true, DATA_FILE_DOCS_DESC); 
    opts.addOption(ExportCEDRParams.DATA_FILE_QUERIES_PARAM, null, true, DATA_FILE_QUERIES_DESC); 
    opts.addOption(ExportCEDRParams.QUERY_DOC_PAIR_FILE_PARAM, null, true, QUERY_DOC_PAIR_FILE_DESC); 
  }
}
