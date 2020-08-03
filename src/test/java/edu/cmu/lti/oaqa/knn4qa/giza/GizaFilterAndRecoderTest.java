/*
 *  Copyright 2015 Carnegie Mellon University
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

import static org.junit.Assert.*;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.lang.exception.ExceptionUtils;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndexBasedFilterAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaTranTableReaderAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.AbstractTest;

/**
 * @author Leonid Boytsov
 */
public class GizaFilterAndRecoderTest extends AbstractTest {
  final static Logger mLogger = LoggerFactory.getLogger(GizaFilterAndRecoderTest.class);
  final static String VOC_SRC_FILE_NAME = "testdata/test_tran/source.vcb";
  final static String VOC_TRG_FILE_NAME = "testdata/test_tran/target.vcb";
  final static String VOC_SRC_FILTER_FILE_NAME = "testdata/test_tran/source_filter.vcb";
  final static String VOC_TRG_FILTER_FILE_NAME = "testdata/test_tran/target_filter.vcb";
  final static String TRAN_TABLE_FILE_NAME = "testdata/test_tran/output.t1";
  final static String FWD_INDEX_NAME = "testdata/test_tran/memfwdindex_Text_bm25";
  
  /**
   * Testing if we can retrieve correct source-word probability by string representation of a word.
   * 
   * @throws Exception
   */
  @Test
  public void testWordProbSrcWord() throws Exception {
    ForwardIndexBasedFilterAndRecoder filterAndRecoder 
    = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

    GizaTranTableReaderAndRecoder tranReader = 
        new GizaTranTableReaderAndRecoder(
                               false, TRAN_TABLE_FILE_NAME, 
                               filterAndRecoder,
                               new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                               new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                               0.5f, 0.0f);

    try {
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb("find"), .99918709200640472964, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb("oddball"), .00029560290676191649, 1e-7));
      // ickiness will be filtered out
//      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb("ickiness"), .00012316787781746520, 1e-7));            
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb("thumbnail"), .00039413720901588865, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }
  
  /**
   * Testing if we can retrieve correct source-word probability by string representation of a word
   * in a flipped translation table.
   * 
   * @throws Exception
   */
  @Test
  public void testWordProbSrcWordFlipped() throws Exception {
    ForwardIndexBasedFilterAndRecoder filterAndRecoder 
    = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

    GizaTranTableReaderAndRecoder tranReader = 
        new GizaTranTableReaderAndRecoder(
                               true, // Flipped table! 
                               TRAN_TABLE_FILE_NAME, 
                               filterAndRecoder,
                               new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                               new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                               0.5f, 0.0f);

    try {
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb("find"), .99918709200640472964, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb("oddball"), .00029560290676191649, 1e-7));
      // ickiness will be filtered out
//      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb("ickiness"), .00012316787781746520, 1e-7));            
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb("thumbnail"), .00039413720901588865, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }  

  /**
   * Testing if we can retrieve correct source-word probability by a recoded word ID.
   * 
   * @throws Exception
   */
  @Test
  public void testWordProbSrcWordId() throws Exception {
    ForwardIndexBasedFilterAndRecoder filterAndRecoder 
    = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

    GizaTranTableReaderAndRecoder tranReader = 
        new GizaTranTableReaderAndRecoder(false,
                               TRAN_TABLE_FILE_NAME, 
                               filterAndRecoder,
                               new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                               new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                               0.5f, 0.0f);

    try {
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb(975), .99918709200640472964, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb(1770), .00029560290676191649, 1e-7));
      // ickiness will be filtered out
//      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb("ickiness"), .00012316787781746520, 1e-7));            
      assertTrue(approxEqual(mLogger, tranReader.getSourceWordProb(2599), .00039413720901588865, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }

  /**
   * Testing if we can retrieve correct source-word probability by a recoded word ID (in a flipped
   * table).
   * 
   * @throws Exception
   */
  @Test
  public void testWordProbSrcWordIdFlipped() throws Exception {
    ForwardIndexBasedFilterAndRecoder filterAndRecoder 
    = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

    GizaTranTableReaderAndRecoder tranReader = 
        new GizaTranTableReaderAndRecoder(
                               true, // flipped!
                               TRAN_TABLE_FILE_NAME, 
                               filterAndRecoder,
                               new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                               new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                               0.5f, 0.0f);

    try {
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb(975), .99918709200640472964, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb(1770), .00029560290676191649, 1e-7));
      // ickiness will be filtered out
//      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb("ickiness"), .00012316787781746520, 1e-7));            
      assertTrue(approxEqual(mLogger, tranReader.getTargetWordProb(2599), .00039413720901588865, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }
  
  /**
   * Testing if we can retrieve correct translation probability for a pair of words.
   * 
   * @throws Exception
   */
  @Test
  public void testTranTableByWords() throws Exception {

    try {
      ForwardIndexBasedFilterAndRecoder filterAndRecoder 
      = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

      GizaTranTableReaderAndRecoder tranReader = 
          new GizaTranTableReaderAndRecoder(false,
                                 TRAN_TABLE_FILE_NAME, 
                                 filterAndRecoder,
                                 new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                                 new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                                 0.5f, 0.0f);
      
      // Filtered-out ickiness

      assertTrue(approxEqual(mLogger, tranReader.getTranProb("find", "find"),           0.562618, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("oddball", "oddball"),     0.5, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("oddball", "lane"),        0.00813415, 1e-7));
      // self-probability will be 0.5f even if though the word is filtered out
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("ickiness", "ickiness"),   0.5, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("thumbnail", "thumbnail"), 0.51827235, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("thumbnail", "same"),      0.0132897, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("thumbnail", "suggest"),   0.00054511, 1e-7));
      
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("testword", "obtain"),   0.05, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("testword", "typical"),   0.1, 1e-7));
    
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }  

  /**
   * Testing if we can retrieve correct translation probability for a pair of words (in a flipped table).
   * 
   * @throws Exception
   */
  @Test
  public void testTranTableByWordsFlipped() throws Exception {

    try {
      ForwardIndexBasedFilterAndRecoder filterAndRecoder 
      = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

      GizaTranTableReaderAndRecoder tranReader = 
          new GizaTranTableReaderAndRecoder(
                                 true, // flipped!
                                 TRAN_TABLE_FILE_NAME, 
                                 filterAndRecoder,
                                 new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                                 new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                                 0.5f, 0.0f);
      
      // Filtered-out ickiness

      assertTrue(approxEqual(mLogger, tranReader.getTranProb("find", "find"),           0.562618, 1e-7));      
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("oddball", "oddball"),     0.5, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("lane", "oddball"),        0.00813415, 1e-7));
      // self-probability will be 0.5f even if though the word is filtered out
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("ickiness", "ickiness"),   0.5, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("thumbnail", "thumbnail"), 0.51827235, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("same", "thumbnail"),      0.0132897, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("suggest", "thumbnail"),   0.00054511, 1e-7));
      
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("obtain", "testword"),   0.05, 1e-7));
      assertTrue(approxEqual(mLogger, tranReader.getTranProb("typical", "testword"),   0.1, 1e-7));
    
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }  
  
  /**
   * Testing if we can retrieve correct translation probability for a pair of words <b>represented by recoded IDs</b>.
   * 
   * @throws Exception
   */
  @Test
  public void testTranTableById() throws Exception {

    try {
      ForwardIndexBasedFilterAndRecoder filterAndRecoder 
      = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

      GizaTranTableReaderAndRecoder tranReader = 
          new GizaTranTableReaderAndRecoder(false,
                                 TRAN_TABLE_FILE_NAME, 
                                 filterAndRecoder,
                                 new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                                 new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                                 0.5f, 0.0f);
      
      // Filtered-out ickiness

      // "find", "find"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(975, 975),     0.562618, 1e-7));      
      // "oddball", "oddball"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(1770, 1770),   0.5, 1e-7));
      // "oddball", "lane"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(1770, 1428),   0.00813415, 1e-7));
      // "thumbnail", "thumbnail"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2599, 2599),   0.51827235, 1e-7));
      // "thumbnail", "same"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2599, 2224),   0.0132897, 1e-7));
      // "thumbnail", "suggest"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2599, 2488),   0.00054511, 1e-7));
      // "testword", "obtain"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2568, 1765),   0.05, 1e-7));
      // "testword", "typical"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2568, 2674),   0.1, 1e-7));

      // Spurious insertions aren't supported any more
      // 0 -> find
      //assertTrue(approxEqual(mLogger, tranReader.getTranProb(0, 975),     0.5, 1e-7));
      // 0 -> list
      //assertTrue(approxEqual(mLogger, tranReader.getTranProb(0, 1484),    0.5, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }  
  
  
  /**
   * Testing if we can retrieve correct translation probability for a pair of words <b>represented by recoded IDs</b>
   * (in a flipped translation table).
   * 
   * @throws Exception
   */
  @Test
  public void testTranTableByIdFlipped() throws Exception {

    try {
      ForwardIndexBasedFilterAndRecoder filterAndRecoder 
      = new ForwardIndexBasedFilterAndRecoder(ForwardIndex.createReadInstance(FWD_INDEX_NAME));

      GizaTranTableReaderAndRecoder tranReader = 
          new GizaTranTableReaderAndRecoder(true, // flipped
                                 TRAN_TABLE_FILE_NAME, 
                                 filterAndRecoder,
                                 new GizaVocabularyReader(VOC_SRC_FILE_NAME, null /* no filter */),
                                 new GizaVocabularyReader(VOC_TRG_FILE_NAME, null /* no filter */),
                                 0.5f, 0.0f);
      
      // Filtered-out ickiness

      // "find", "find"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(975, 975),     0.562618, 1e-7));      
      // "oddball", "oddball"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(1770, 1770),   0.5, 1e-7));
      // "oddball", "lane"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(1428, 1770),   0.00813415, 1e-7));
      // "thumbnail", "thumbnail"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2599, 2599),   0.51827235, 1e-7));
      // "thumbnail", "same"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2224, 2599),   0.0132897, 1e-7));
      // "thumbnail", "suggest"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2488, 2599),   0.00054511, 1e-7));
      // "testword", "obtain"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(1765, 2568),   0.05, 1e-7));
      // "testword", "typical"
      assertTrue(approxEqual(mLogger, tranReader.getTranProb(2674, 2568),   0.1, 1e-7));

      /* Spurious insertions aren't supported any more */
      // 0 -> find
      //assertTrue(approxEqual(mLogger, tranReader.getTranProb(975, 0),     0.5, 1e-7));
      // 0 -> list
      //assertTrue(approxEqual(mLogger, tranReader.getTranProb(1484, 0),    0.5, 1e-7));
    } catch (Exception e) {
      mLogger.error(ExceptionUtils.getFullStackTrace(e));
      fail();
    }
  }  

  
}
