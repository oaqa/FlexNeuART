package edu.cmu.lti.oaqa.flexneuart.apps;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import edu.cmu.lti.oaqa.flexneuart.utils.RandomUtils;

public abstract class ExportTrainNegSampleBase extends ExportTrainBase {
  static final Logger logger = LoggerFactory.getLogger(ExportTrainNegSampleBase.class);

  private static final String MAX_HARD_NEG_QTY = "hard_neg_qty";
  private static final String MAX_SAMPLE_MEDIUM_NEG_QTY = "sample_med_neg_qty";
  private static final String MAX_SAMPLE_EASY_NEG_QTY = "sample_easy_neg_qty";
  private static final String MAX_CAND_TRAIN_QTY_PARAM = "cand_train_qty";
  private static final String MAX_CAND_TRAIN_QTY_DESC = "A max. # of candidate records from which we sample medium negatives.";
  private static final String MAX_CAND_TRAIN_FOR_POS_QTY_PARAM = "cand_train_4pos_qty";
  private static final String MAX_CAND_TRAIN_FOR_POS_QTY_DESC = "A max. # of candidate records from which we select positive samples.";
  private static final String MAX_CAND_TEST_QTY_PARAM = "cand_test_qty";
  private static final String MAX_CAND_TEST_QTY_DESC = "max # of candidate records returned by the provider to generate test data.";
  public static final String MAX_DOC_WHITESPACE_TOK_QTY_PARAM = "max_doc_whitespace_qty";
  public static final String MAX_DOC_WHITESPACE_TOK_QTY_DESC = "max # of whitespace separated tokens to keep in a document";
  public static final String KEEP_CASE_PARAM = "keep_case";
  public static final String KEEP_CASE_DESC = "do not lower-case queries and documents";
  
  public ExportTrainNegSampleBase(ForwardIndex fwdIndex, String queryExportFieldName, String indexExportFieldName,
      QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(fwdIndex, queryExportFieldName, indexExportFieldName, qrelsTrain, qrelsTest);
    
    mAllDocIds = fwdIndex.getAllDocIds();
  }

  protected static void addOptionsDesc(Options opts) {
    opts.addOption(MAX_HARD_NEG_QTY, null, true, "A max. # of *HARD* negative examples (all K top-score candidates) per query");
    opts.addOption(MAX_SAMPLE_MEDIUM_NEG_QTY, null, true, "A max. # of *MEDIUM* negative samples (negative candidate and QREL samples) per query");
    opts.addOption(MAX_SAMPLE_EASY_NEG_QTY, null, true, "A max. # of *EASY* negative samples (sampling arbitrary docs) per query");
    
    opts.addOption(MAX_CAND_TRAIN_QTY_PARAM, null, true, MAX_CAND_TRAIN_QTY_DESC);
    opts.addOption(MAX_CAND_TRAIN_FOR_POS_QTY_PARAM, null, true, MAX_CAND_TRAIN_FOR_POS_QTY_DESC);
    opts.addOption(MAX_CAND_TEST_QTY_PARAM, null, true, MAX_CAND_TEST_QTY_DESC);
    opts.addOption(CommonParams.RANDOM_SEED_PARAM, null, true, CommonParams.RANDOM_SEED_DESC);
    
    opts.addOption(MAX_DOC_WHITESPACE_TOK_QTY_PARAM, null, true, MAX_DOC_WHITESPACE_TOK_QTY_DESC);
    
    opts.addOption(KEEP_CASE_PARAM, null, false, KEEP_CASE_DESC);
  }

  protected int mHardNegQty = 0;

  @Override
  protected String readAddOptions(CommandLine cmd) {
    
    // Only this sampling parameter should be mandatory
    String tmpn = cmd.getOptionValue(MAX_SAMPLE_MEDIUM_NEG_QTY);
    if (null == tmpn) {
      return "Specify option: " + MAX_SAMPLE_MEDIUM_NEG_QTY;
    }
    try {
      mSampleMedNegQty = Math.max(0, Integer.parseInt(tmpn));
    } catch (NumberFormatException e) {
      return MAX_SAMPLE_MEDIUM_NEG_QTY + " isn't integer: '" + tmpn + "'";
    }
    
    tmpn = cmd.getOptionValue(MAX_HARD_NEG_QTY);
    if (null != tmpn) {
      try {
        mHardNegQty = Math.max(0, Integer.parseInt(tmpn));
      } catch (NumberFormatException e) {
        return MAX_HARD_NEG_QTY + " isn't integer: '" + tmpn + "'";
      }
    }
    
    tmpn = cmd.getOptionValue(MAX_SAMPLE_EASY_NEG_QTY);
    if (null != tmpn) {
      try {
        mSampleEasyNegQty = Math.max(0, Integer.parseInt(tmpn));
      } catch (NumberFormatException e) {
        return MAX_SAMPLE_EASY_NEG_QTY + " isn't integer: '" + tmpn + "'";
      }
    }
    
    logger.info(String.format("# of hard/medium/easy samples per query: %d/%d/%d", mHardNegQty, mSampleMedNegQty, mSampleEasyNegQty));
    logger.info("# of candidates to select relevant entries (for exporting with scores only)" + this.mCandTrain4PosQty);
    
    tmpn = cmd.getOptionValue(MAX_CAND_TRAIN_QTY_PARAM);
    if (null != tmpn) {
      try {
        mCandTrainQty = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return MAX_CAND_TRAIN_QTY_PARAM + " isn't integer: '" + tmpn + "'";
      }
    }
    
    mCandTrain4PosQty = mCandTrainQty;
    tmpn = cmd.getOptionValue(MAX_CAND_TRAIN_FOR_POS_QTY_PARAM);
    if (null != tmpn) {
      try {
        mCandTrainQty = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return MAX_CAND_TRAIN_FOR_POS_QTY_PARAM + " isn't integer: '" + tmpn + "'";
      }
    }
    
    
    int seed = 0;
    tmpn = cmd.getOptionValue(CommonParams.RANDOM_SEED_PARAM);
    if (null != tmpn) {
      try {
        seed = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return CommonParams.RANDOM_SEED_PARAM + " isn't integer: '" + seed + "'";
      }
    }
    
    mRandUtils = new RandomUtils(seed);
     
    tmpn = cmd.getOptionValue(MAX_CAND_TEST_QTY_PARAM);
    if (null != tmpn) {
      try {
        mCandTestQty = Integer.parseInt(tmpn);
      } catch (NumberFormatException e) {
        return MAX_CAND_TEST_QTY_PARAM + " isn't integer: '" + tmpn + "'";
      }
    }
    
    if (cmd.hasOption(KEEP_CASE_PARAM)) {
      mDoLowerCase = false;
    }
    
    logger.info("Lower-casing? " + mDoLowerCase);
    
    logger.info(String.format("# top-scoring training candidates to sample/select from %d", mCandTrainQty));
    logger.info(String.format("# top candidates for validation %d", mCandTestQty));
    
    
    tmpn = cmd.getOptionValue(MAX_DOC_WHITESPACE_TOK_QTY_PARAM);
    if (tmpn != null) {
      try {
        mMaxWhitespaceTokDocQty = Integer.parseInt(tmpn);
      }  catch (NumberFormatException e) {
        return "Maximum number of whitespace document tokens isn't integer: '" + tmpn + "'";
      }
    }
    if (mMaxWhitespaceTokDocQty > 0) {
      logger.info(String.format("Keeping max %d number of whitespace document tokens", mMaxWhitespaceTokDocQty));
    }
    
    return "";
  }

  @Override
  protected abstract void startOutput() throws Exception;

  @Override
  protected abstract void finishOutput() throws Exception;

  abstract void exportQueryTest(CandidateProvider candProv,
                                int queryNum,
                                DataEntryFields queryEntry,
                                String queryExportFieldText) throws Exception;
  
  abstract void exportQueryTrain(CandidateProvider candProv,
                                int queryNum,
                                DataEntryFields queryEntry,
                                String queryExportFieldText) throws Exception;
  
  /**
   *  A wrapper functions that routes the query to a specific processing function and does case-processing (lower-casing)
   *  if needed.
   *  
   *  @param candProv    candidate provider
   *  @param queryNum    a query number
   *  @param queryEntry  a query data entry {@link edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields}
   */
  @Override
  protected void exportQuery(CandidateProvider candProv, int queryNum, DataEntryFields queryEntry, boolean bIsTestQuery) throws Exception {
    String queryExportFieldText = handleCase(queryEntry.getStringDefault(mQueryExportFieldName, "")).trim();
    if (bIsTestQuery) {
      exportQueryTest(candProv, queryNum, queryEntry, queryExportFieldText);
    } else {
      exportQueryTrain(candProv, queryNum, queryEntry, queryExportFieldText);
    }
  }

  protected int mSampleMedNegQty = 0;
  protected int mSampleEasyNegQty = 0;
  protected int mMaxWhitespaceTokDocQty = -1;
  protected int mCandTrainQty = Integer.MAX_VALUE;
  protected int mCandTrain4PosQty = Integer.MAX_VALUE;
  protected int mCandTestQty = Integer.MAX_VALUE;
  protected RandomUtils mRandUtils = null;
  protected String [] mAllDocIds = null;

}