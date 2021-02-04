package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.opencsv.CSVWriter;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.utils.CompressUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

class ExportTrainMatchZoo extends ExportTrainNegSampleBase {
  

  public static final String OUTPUT_FILE_TRAIN_PARAM = "out_file_train";
  public static final String OUTPUT_FILE_TRAIN_DESC = "Output file for training data";
  
  public static final String OUTPUT_FILE_TEST_PARAM = "out_file_test";
  public static final String OUTPUT_FILE_TEST_DESC = "Output file for test data";
  
  private static final Logger logger = LoggerFactory.getLogger(ExportTrainMatchZoo.class);
  public static final String FORMAT_NAME = "match_zoo";
  
  protected ExportTrainMatchZoo(ForwardIndex fwdIndex, 
                               QrelReader qrelsTrain, QrelReader qrelsTest) {
    super(fwdIndex, qrelsTrain, qrelsTest);
  }

  // Must be called from ExportTrainBase.addAllOptionDesc
  static void addOptionDesc(Options opts) {
    opts.addOption(OUTPUT_FILE_TRAIN_PARAM, null, true, OUTPUT_FILE_TRAIN_DESC); 
    opts.addOption(OUTPUT_FILE_TEST_PARAM, null, true, OUTPUT_FILE_TEST_DESC); 
  }

  @Override
  String readAddOptions(CommandLine cmd) {
    String err = super.readAddOptions(cmd);
    if (!err.isEmpty()) {
      return err;
    }
    
    mOutFileNameTrain = cmd.getOptionValue(OUTPUT_FILE_TRAIN_PARAM);    
    if (null == mOutFileNameTrain) {
      return "Specify option: " + OUTPUT_FILE_TRAIN_PARAM;
    }
    mOutFileNameTest = cmd.getOptionValue(OUTPUT_FILE_TEST_PARAM);    
    if (null == mOutFileNameTest) {
      return "Specify option: " + OUTPUT_FILE_TEST_PARAM;
    }    

    return "";
  }
  
  synchronized void writeField(CSVWriter out,
                               String idLeft, String textLeft,
                               String idRight, String textRight,
                               int relFlag) throws Exception {
    
    String lineFields[] = { idLeft, textLeft, idRight, textRight, "" + relFlag};
    out.writeNext(lineFields);
    mOutNum++;    
  }

  @Override
  void startOutput() throws Exception {
    String lineFields[] = {"id_left", "text_left", "id_right", "text_right", "label"};
    
    mOutTrain = new CSVWriter(new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(mOutFileNameTrain))),
                        ',', // field separator
                        CSVWriter.NO_QUOTE_CHARACTER, // quote char
                        CSVWriter.NO_ESCAPE_CHARACTER, // escape char
                        Const.NL
                        );
    mOutTrain.writeNext(lineFields);
    mOutTest = new CSVWriter(new BufferedWriter(new OutputStreamWriter(CompressUtils.createOutputStream(mOutFileNameTest))),
        ',', // field separator
        CSVWriter.NO_QUOTE_CHARACTER, // quote char
        CSVWriter.NO_ESCAPE_CHARACTER, // escape char
        Const.NL
        );
    mOutTest.writeNext(lineFields);
    
    mOutNum = 0;
  }

  @Override
  void finishOutput() throws Exception {
    logger.info("Generated data for " + mOutNum + " query-doc pairs.");
    mOutTrain.close();
    mOutTest.close();
  }
  
  @Override
  void writeOneEntryData(String queryFieldText, boolean isTestQuery,
                         String queryId,
                         HashSet<String> relDocIds, ArrayList<String> docIds) throws Exception {
    for (String docId : docIds) {
      int relFlag = relDocIds.contains(docId) ? 1 : 0;
      
      String text = getDocText(docId);
      
      if (text == null) {
        logger.warn("Ignoring document " + docId + " b/c of null field");
        continue;
      }
      
      if (text.isEmpty()) {
        logger.warn("Ignoring document " + docId + " b/c of empty field");
        continue;
      }
      
      if (mMaxWhitespaceTokDocQty > 0) {
        text = StringUtils.truncAtKthWhiteSpaceSeq(text, mMaxWhitespaceTokDocQty);
      }
      
      writeField(isTestQuery ? mOutTest : mOutTrain, queryId, queryFieldText, docId, text, relFlag);
    }
  }
  
  int                   mOutNum = 0;
  
  CSVWriter             mOutTrain;
  CSVWriter             mOutTest;

  String                 mOutFileNameTrain;
  String                 mOutFileNameTest;
  
}