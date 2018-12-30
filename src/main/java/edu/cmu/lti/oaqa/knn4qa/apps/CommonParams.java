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
package edu.cmu.lti.oaqa.knn4qa.apps;

import edu.cmu.lti.oaqa.knn4qa.letor.InMemIndexFeatureExtractorOld;

public class CommonParams {
  
  public final static String LUCENE_INDEX_LOCATION_DESC = "Location of a Lucene index";
  public static final String PROVIDER_URI_DESC = "Provider URI: an index location or a query server address";
  public static final String PROVIDER_URI_PARAM = "u";
  
  public static final String QUERY_CACHE_FILE_DESC  = "A file to cache query results";
  public static final String QUERY_CACHE_FILE_PARAM = "query_cache_file";
  
  public static final String QUERY_FILE_DESC = "Query file";
  public static final String QUERY_FILE_PARAM = "q";
  
  public static final String QREL_FILE_DESC  = "QREL file: "+ 
                          " if specified, we save results only for queries for which we find at least one relevant entry.";
  public static final String QREL_FILE_PARAM = "qrel_file";
  
  public static final String MODEL_FILE_FINAL_PARAM = "model_final";
  public static final String MODEL_FILE_FINAL_DESC = "Final stage learning-to-rank model";

  public static final String MODEL_FILE_INTERM_PARAM = "model_interm";
  public static final String MODEL_FILE_INTERM_DESC = "Intermediate learning-to-rank model";  
  
  public static final String FEATURE_FILE_PARAM = "f";
  public static final String FEATURE_FILE_DESC  = "An output file for features in the SVM-rank format";
  
  public static final String TREC_STYLE_OUT_FILE_PARAM = "o";
  public static final String TREC_STYLE_OUT_FILE_DESC = "A TREC-style (QRELs) output file";
  
  public final static String THREAD_QTY_DESC   = "The number of threads";
  public final static String THREAD_QTY_PARAM  = "thread_qty";
  
  public final static String KNN_THREAD_QTY_DESC   = "The number of threads of knn brute-force candidate provider";
  public final static String KNN_THREAD_QTY_PARAM  = "knn_thread_qty";
  
  public final static String MEMINDEX_DESC = "A directory for in-memory index";
  public final static String MEMINDEX_PARAM = "memindex_dir";
    
  public static final String MEM_FWD_INDEX_PARAM = "memindex";
  public static final String MEM_FWD_INDEX_DESC = "A forward index file used for filtering";

  public final static String CAND_PROVID_DESC = "candidate record provider type, e.g., lucene, nmslib";
  public final static String CAND_PROVID_PARAM = "cand_prov";
  
  public final static String MIN_SHOULD_MATCH_PCT_PARAM = "min_should_match_pct";
  public final static String MIN_SHOULD_MATCH_PCT_DESC  = "a percentage of query word (an integer from 0 to 100) that must match a document word";

  public final static String EXTRACTOR_TYPE_FINAL_PARAM = "extr_type_final";
  public final static String EXTRACTOR_TYPE_FINAL_DESC = "Final-stage extrator type: " + 
                                                  InMemIndexFeatureExtractorOld.getExtractorListDesc();
                                  ;
  public final static String EXTRACTOR_TYPE_INTERM_PARAM = "extr_type_interm";  
  public final static String EXTRACTOR_TYPE_INTERM_DESC = "Intermediate extrator type: " + 
                                                  InMemIndexFeatureExtractorOld.getExtractorListDesc();
  
  public final static String MAX_CAND_QTY_PARAM    = "cand_qty";
  public final static String MAX_CAND_QTY_DESC     = "A maximum number of candidate records returned by the provider. " +
                                                     "This is used only in conjunction with an intermediate re-ranker.";
  
  public final static String MAX_NUM_RESULTS_PARAM = "n";
  public final static String MAX_NUM_RESULTS_DESC  = "A comma-separated list of numbers of candidate records (per-query).";
  
  public static final String GIZA_ROOT_DIR_PARAM = "giza_root_dir";
  public static final String GIZA_ROOT_DIR_DESC =  "a root dir for GIZA output";
    
  public static final String GIZA_ITER_QTY_PARAM = "giza_iter_qty";
  public static final String GIZA_ITER_QTY_DESC = "a number of GIZA iterations";

  public static final String GIZA_EXPAND_QTY_PARAM = "giza_expand_qty";
  public static final String GIZA_EXPAND_QTY_DESC  = "A number of GIZA-based query-expansion terms";
  
  public static final String GIZA_EXPAND_USE_WEIGHTS_PARAM = "giza_wght_expand";
  public static final String GIZA_EXPAND_USE_WEIGHTS_DESC = "Use translation probabilities as weights during expansion";  
  
  public static final String EMBED_DIR_PARAM = "embed_dir";
  public static final String EMBED_DIR_DESC = "a root dir for word embeddings";
  
  public static final String EMBED_FILES_PARAM = "embed_files";
  public static final String EMBED_FILES_DESC  = "a comma-separated list of word embedding file names";
  
  public static final String HIHG_ORDER_FILES_PARAM = "horder_files";
  public static final String HIHG_ORDER_FILES_DESC  = "a comma-separated list of sparse (high-order models) word embedding file names";
  
  public final static String KNN_WEIGHTS_FILE_DESC = "a file with the weights for knn-search";
  public final static String KNN_WEIGHTS_FILE_PARAM = "knn_weights";
  
  public final static String KNN_QUERIES_DESC  = "a file to save knn-queries in the format that can be processed by NMSLIB";
  public final static String KNN_QUERIES_PARAM = "knn_queries";    
  
  public final static String KNN_SERVICE_PARAM = "nmslib_addr";
  public final static String KNN_SERVICE_DESC  = "the address (in the format host:port) of the NMSLIB server";
  
  public final static String NMSLIB_FIELDS_PARAM = "nmslib_fields";
  public final static String NMSLIB_FIELDS_DESC  = "A comma-separated list of fields used by an NMSLIB provider, it must correspond exactly to what is specified in respective NMSLIB header file";
  
  public final static String ROOT_DIR_DESC = "A root dir for the pipeline output";
  public final static String ROOT_DIR_PARAM = "root_dir";
    
  public final static String SUB_DIR_TYPE_DESC = "A coma separated list of directories to be included, e.g., train,dev1,dev2,test";
  public final static String SUB_DIR_TYPE_PARAM = "sub_dirs";
    
  public final static String MAX_NUM_REC_DESC = "maximum number of records to process";
  public final static String MAX_NUM_REC_PARAM = "n";
  
  public final static String MAX_NUM_QUERY_DESC  = "maximum number of queries to process";
  public final static String MAX_NUM_QUERY_PARAM = "max_num_query";
  
  public final static String SOLR_FILE_NAME_DESC = "A name of output file to be fed to a SOLR indexer, e.g., SolrAnswerFile.txt";
  public final static String SOLR_FILE_NAME_PARAM = "solr_file";
    
  public final static String OUT_MINDEX_DESC = "A directory to store index";
  public final static String OUT_INDEX_PARAM = "index_dir";
  
  public final static String MIN_PROB_PARAM  = "min_prob";
  public final static String MIN_PROB_DESC   = "A minimum probability";
  
  public final static String SEL_PROB_PARAM  = "sel_prob";
  public final static String SEL_PROB_DESC   = "A selection probability";
      
  public final static String MAX_WORD_QTY_PARAM = "max_word_qty";
  public final static String MAX_WORD_QTY_DESC  = "A maximum number of words";

  public final static String SAVE_STAT_FILE_PARAM = "save_stat_file";
  public final static String SAVE_STAT_FILE_DESC  = "A file to save some vital query execution statistics";
  
  public final static String USE_THREAD_POOL_PARAM = "use_thread_pool";
  public final static String USE_THREAD_POOL_DESC = "Use a thread pool instead of a round-robin division of queries among threads";
  
  public final static String GALAGO_OP_PARAM = "galago_op";
  public final static String GALAGO_OP_DESC  = "A type of retrieval operator, eg., combine, sdm, or rm to use with Galago provider";

  public final static String GALAGO_PARAMS_PARAM = "galago_params";
  public final static String GALAGO_PARAMS_DESC  = "galago_params_desc";
}
