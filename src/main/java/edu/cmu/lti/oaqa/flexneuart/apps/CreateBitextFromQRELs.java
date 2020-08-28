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
package edu.cmu.lti.oaqa.flexneuart.apps;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.embed.EmbeddingReaderAndRecoder;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.letor.FeatExtrResourceManager;
import edu.cmu.lti.oaqa.flexneuart.simil_func.AbstractDistance;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryReader;
import edu.cmu.lti.oaqa.flexneuart.utils.QrelReader;
import edu.cmu.lti.oaqa.flexneuart.utils.RandomUtils;
import edu.cmu.lti.oaqa.flexneuart.utils.StringUtils;

/**
 * Creating parallel (bi-text) corpus from relevance judgements, queries,
 * and indexed documents.
 *
 * Importantly, it requires a forward index to store sequences of words
 * rather than merely bag-of-words entries.
 * 
 * @author Leonid Boytsov
 *
 */
public class CreateBitextFromQRELs {
  public final static String EMBED_FILE_NAME_PARAM = "-embed_file";
  public final static String MAX_DOC_QUERY_QTY_RATIO_PARAM = "-max_doc_query_qty_ratio";
  
  static RandomUtils rand = new RandomUtils(0);
  
  public static final class Args {  
    @Option(name = "-" + CommonParams.FWDINDEX_PARAM, required = true, usage = CommonParams.FWDINDEX_DESC)
    String mFwdIndex;
    @Option(name = "-" + CommonParams.QUERY_FILE_PARAM, required = true, usage = CommonParams.QUERY_FILE_DESC)
    String mQueryFile;
    
    @Option(name = "-" + CommonParams.INDEX_FIELD_NAME_PARAM, required = true, usage = CommonParams.INDEX_FIELD_NAME_DESC)
    String mIndexField;
    @Option(name = "-" + CommonParams.QUERY_FIELD_NAME_PARAM, required = true, usage = CommonParams.INDEX_FIELD_NAME_DESC)
    String mQueryField;
    @Option(name = "-" + CommonParams.QREL_FILE_PARAM, required = true, usage = CommonParams.QREL_FILE_DESC)
    String mQrelFile;
    @Option(name = "-output_dir", required = true, usage = "bi-text output directory")
    String mOutDir;
    @Option(name = "-" + CommonParams.EMBED_DIR_PARAM, usage = CommonParams.EMBED_DIR_DESC)
    String mEmbedDir;
    @Option(name = EMBED_FILE_NAME_PARAM, usage = "embedding file name relative to the root (used for document word sampling)")
    String mEmbedFile;
    @Option(name = "-sample_qty", usage = "number of samples per query, if specified we need embeddings")
    int mSampleQty = -1;
    
    @Option(name = "-" + CommonParams.MAX_NUM_QUERY_PARAM, required = false, usage = CommonParams.MAX_NUM_QUERY_DESC)
    int mMaxNumQuery = Integer.MAX_VALUE;
    
    @Option(name = MAX_DOC_QUERY_QTY_RATIO_PARAM, usage = "Max. ratio of # words in docs to # of words in queries (<=0 to dump complete raw text)")
    float mDocQueryWordRatio = 0;
  }
  
  public static void main(String[] argv) {

    Args args = new Args();
    CmdLineParser parser = null;
    
    try {
 
      parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(CommonParams.USAGE_WIDTH));
      parser.parseArgument(argv);
    
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.exit(1);
    }    
    
    
    try {
      FeatExtrResourceManager resourceManager = new FeatExtrResourceManager(args.mFwdIndex, null, args.mEmbedDir);
      EmbeddingReaderAndRecoder embeds = null;

      
      String fieldName = args.mIndexField;
      
      if (args.mSampleQty > 0) {
        if (args.mEmbedDir == null) {
          System.err.println("For sampling you need to specify: -" + CommonParams.EMBED_DIR_PARAM);
          System.exit(1);
        }
        if (args.mEmbedFile == null) {
          System.err.println("For sampling you need to specify (relative to embedding dir root): " + EMBED_FILE_NAME_PARAM);
          System.exit(1); 
        }
        
        embeds = resourceManager.getWordEmbed(fieldName, args.mEmbedFile);
      }
   
      ForwardIndex fwdIndex = resourceManager.getFwdIndex(fieldName);
      
      QrelReader qrels = new QrelReader(args.mQrelFile);
      
      BufferedWriter questFile = 
          new BufferedWriter(new FileWriter(args.mOutDir + Const.PATH_SEP + Const.BITEXT_QUEST_PREFIX + fieldName));
      BufferedWriter answFile = 
            new BufferedWriter(new FileWriter(args.mOutDir + Const.PATH_SEP + Const.BITEXT_ANSW_PREFIX + fieldName));
          
      Map<String, String> docFields = null;
      int queryQty = 0;
      
      DataEntryReader inp = new DataEntryReader(args.mQueryFile);
      
      for (; ((docFields = inp.readNext()) != null) && queryQty < args.mMaxNumQuery; ) {

        ++queryQty;
        
        String qid = docFields.get(Const.TAG_DOCNO);
        if (qid == null) {
          System.err.println("Undefined query ID in query # " + queryQty);
          System.exit(1);
        }
        
        String queryText = docFields.get(args.mQueryField);
        if (queryText == null) queryText = "";
        queryText = queryText.trim();
        String [] queryWords = StringUtils.splitOnWhiteSpace(queryText);
      
        if (queryText.isEmpty() || queryWords.length == 0) {
          System.out.println("Empty text in query # " + queryQty + " ignoring");
          continue;
        }
        
        float queryWordQtyInv = 1.0f / queryWords.length;
        
        HashMap<String, String> relInfo = qrels.getQueryQrels(qid);
        if (relInfo == null) {
          System.out.println("Warning: no QRELs for query id: " + qid);
          continue;
        }
        for (Entry<String, String> e : relInfo.entrySet()) {
          String did = e.getKey();
          int grade = CandidateProvider.parseRelevLabel(e.getValue());
          if (grade >= 1) {
            if (args.mSampleQty <= 0) {
              if (args.mDocQueryWordRatio > 0)
                genBitextSplitPlain(fwdIndex, did, fieldName, questFile, answFile, queryText, queryWordQtyInv, args.mDocQueryWordRatio);
              else 
                genBitextWholeRaw(fwdIndex, did, fieldName, questFile, answFile, queryText);
            } else {
              genBitextSample(fwdIndex, embeds, did, fieldName, questFile, answFile, queryWords, args.mSampleQty);
            }
          }
        }  
      } 
      
      questFile.close();
      answFile.close();  
 
      inp.close();
      
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Exception: " + e);
      System.exit(1);
    }
     
  }
  
  private static void genBitextSample(ForwardIndex fwdIndex, EmbeddingReaderAndRecoder embeds, 
                                     String did, String fieldName, 
                                     BufferedWriter questFile, BufferedWriter answFile, 
                                     String[] queryWords, int sampleQty) throws Exception {
    if (fwdIndex.isRaw()) {
      throw new Exception("genBitextSample requires a parsed forward index!");
    }
    DocEntryParsed dentry = fwdIndex.getDocEntryParsed(did);
    if (dentry == null) {
      System.out.println("Seems like data inconsistency, there is no document " + did + " index, but there is a QREL entry with it");
      return;
    }

    AbstractDistance dist = AbstractDistance.create(AbstractDistance.COSINE);
    
    int docWordQty = dentry.mWordIds.length;
    if (docWordQty == 0) {
      System.out.println("Empty doc " + did + " for field: " + fieldName);
      return; // emtpy doc
    }
    float weights[] = new float[docWordQty];
    
    ArrayList<ArrayList<String>>  sampledDocWords = new ArrayList<>();
    
    for (int i = 0; i < sampleQty; ++i) {
      sampledDocWords.add(new ArrayList<String>()); // Each sample has an array of words
    }
    
    for (int iq = 0; iq < queryWords.length; ++iq) {
      float[] qvec = embeds.getVector(queryWords[iq]);
      
      float weightSum = 0;
      if (qvec != null) {
        for (int iWord = 0; iWord < docWordQty; ++iWord) {
          int wordId = dentry.mWordIds[iWord];
          float[] dvec = embeds.getVector(wordId);
          if (dvec != null) {
            // A sample weight can be proportional to both the similarity and the # of occurrences,
            // but let's dampen it a bit to give other words a chance:
            float w = (float)Math.sqrt((1 + dist.compute(qvec, dvec)) * dentry.mQtys[iWord]);
            // note that all weights have to be non-negative!
            weights[iWord] = w;
            weightSum += w;
          } else {
            weights[iWord] = 0;
          }
        } 
        if (weightSum > 0) {
          int sampledWordIdx[] = rand.sampleWeightWithReplace(weights, sampleQty);
          for (int sampleId = 0; sampleId < sampleQty; ++sampleId) {
            int wid = dentry.mWordIds[sampledWordIdx[sampleId]];
            sampledDocWords.get(sampleId).add(fwdIndex.getWord(wid));
          }
        }
      }
    }
    
    for (int i = 0; i < sampleQty; ++i) {
      ArrayList<String> answWords = sampledDocWords.get(i);
      if (!answWords.isEmpty()) {
          questFile.write(StringUtils.joinWithSpace(queryWords) + Const.NL);
          answFile.write(StringUtils.joinWithSpace(answWords) + Const.NL);
      }
    }
  }

  static void genBitextSplitPlain(ForwardIndex fwdIndex, String did, String fieldName, 
                          BufferedWriter questFile, BufferedWriter answFile,
                          String queryText, 
                          float queryWordQtyInv, float docQueryWordRatio) throws Exception {
    if (fwdIndex.isRaw()) {
      throw new Exception("genBitextSplitPlain requires a parsed forward index!");
    }
    DocEntryParsed dentry = fwdIndex.getDocEntryParsed(did);
    if (dentry == null) {
      System.out.println("Seems like data inconsistency, there is no document " + did + " index, but there is a QREL entry with it");
      return;
    }
    if (dentry.mWordIdSeq == null) {
      System.err.println("Index for the field " + fieldName + " doesn't have words sequences!");
      System.exit(1);
    }      

    ArrayList<String> answWords = new ArrayList<String>();
    /*
     * In principle, queries can be longer then documents, especially,
     * when we write the "tail" part of the documents. However, the 
     * difference should not be large and longer queries will be 
     * truncated by  
     */
    for (int wid : dentry.mWordIdSeq) {
      if (answWords.size() * queryWordQtyInv >= docQueryWordRatio) {
        questFile.write(queryText + Const.NL);
        answFile.write(StringUtils.joinWithSpace(answWords) + Const.NL);
        answWords.clear();
      }
      
      if (wid >=0) { // -1 is OOV
        answWords.add(fwdIndex.getWord(wid));
      }
    }
    
    if (!answWords.isEmpty()) {
      questFile.write(queryText + Const.NL);
      answFile.write(StringUtils.joinWithSpace(answWords) + Const.NL);
    }
  }

  static void genBitextWholeRaw(ForwardIndex fwdIndex, String did, String fieldName, 
      BufferedWriter questFile, BufferedWriter answFile,
      String queryText) throws Exception {
    if (!fwdIndex.isRaw()) {
      throw new Exception("genBitextWholeRaw requires a raw forward index!");
    }
    String docTextRaw = fwdIndex.getDocEntryRaw(did);
    if (docTextRaw == null) {
      System.out.println("Seems like data inconsistency, there is no document " + did + " index, but there is a QREL entry with it");
      return;
    }
    
    docTextRaw = docTextRaw.trim();
    
    if (!docTextRaw.isEmpty()) {
      questFile.write(queryText + Const.NL);
      answFile.write(docTextRaw + Const.NL);
    }
  }  
  
}
