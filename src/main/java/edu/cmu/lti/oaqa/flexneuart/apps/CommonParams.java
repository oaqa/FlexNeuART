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
package edu.cmu.lti.oaqa.flexneuart.apps;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;

public class CommonParams {
  
  public final static String LUCENE_INDEX_LOCATION_DESC = "Location of a Lucene index";
  
  public static final String RUN_ID_PARAM = "run_id";
  public static final String RUN_ID_DESC = "a trec-style run id";
  
  public static final String RANDOM_SEED_PARAM = "seed";
  public static final String RANDOM_SEED_DESC = "a random seed";
  
  public static final String PROVIDER_URI_DESC = "Provider URI: an index location, a query server address, etc";
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
  public static final String TREC_STYLE_OUT_FILE_DESC = "A TREC-style output file";
  
  public final static String THREAD_QTY_DESC   = "The number of threads";
  public final static String THREAD_QTY_PARAM  = "thread_qty";
 
  public final static String FWDINDEX_PARAM = "fwd_index_dir";
  public final static String FWDINDEX_DESC = "A forward-index directory";
 
  public static final String FLT_FWD_INDEX_PARAM = "flt_fwd_index_header";
  public static final String FLT_FWD_INDEX_DESC = "A forward index header file used for filtering";

  public final static String CAND_PROVID_DESC = "candidate record provider type: " + CandidateProvider.CAND_PROVID_DESC;
  public final static String CAND_PROVID_PARAM = "cand_prov";
  
  public final static String CAND_PROVID_ADD_CONF_DESC = "JSON with additional candidate provider parameters";
  public final static String CAND_PROVID_ADD_CONF_PARAM = "cand_prov_add_conf";

  public final static String EXTRACTOR_TYPE_FINAL_PARAM = "extr_type_final";
  public final static String EXTRACTOR_TYPE_FINAL_DESC = "Final-stage extrator type/json"; 
                                  
  public final static String EXTRACTOR_TYPE_INTERM_PARAM = "extr_type_interm";  
  public final static String EXTRACTOR_TYPE_INTERM_DESC = "Intermediate extrator type/json";
  
  public final static String MAX_CAND_QTY_PARAM    = "cand_qty";
  public final static String MAX_CAND_QTY_DESC     = "A maximum number of candidate records returned by the provider. ";
  
  public final static String MAX_FINAL_RERANK_QTY_PARAM    = "max_final_rerank_qty";
  public final static String MAX_FINAL_RERANK_QTY_DESC     = "A maximum number of records to re-rank using the final re-ranker (candidate or re-ranked by the intermediate re-ranker) ";
  
  public final static String MAX_NUM_RESULTS_PARAM = "n";
  public final static String MAX_NUM_RESULTS_DESC  = "A comma-separated list of numbers of candidate records (per-query).";
  
  public static final String GIZA_ROOT_DIR_PARAM = "giza_root_dir";
  public static final String GIZA_ROOT_DIR_DESC =  "a root dir for GIZA output";
    
  public static final String GIZA_ITER_QTY_PARAM = "giza_iter_qty";
  public static final String GIZA_ITER_QTY_DESC = "a number of GIZA iterations";
 
  public static final String EMBED_DIR_PARAM = "embed_dir";
  public static final String EMBED_DIR_DESC = "a root dir for embeddings";

  public final static String INPUT_DATA_DIR_DESC = "A data directory (to be used for indexing/querying)";
  public final static String INPUT_DATA_DIR_PARAM = "input_data_dir";
    
  public final static String INPDATA_SUB_DIR_TYPE_DESC = "A coma separated list of data sub-directories to be used, e.g., train,dev1,dev2,test";
  public final static String INPDATA_SUB_DIR_TYPE_PARAM = "data_sub_dirs";
    
  public final static String MAX_NUM_REC_DESC = "maximum number of records to process";
  public final static String MAX_NUM_REC_PARAM = "n";
  
  public final static String MAX_NUM_QUERY_DESC  = "maximum number of queries to process";
  public final static String MAX_NUM_QUERY_PARAM = "max_num_query";
  
  public final static String MAX_NUM_QUERY_TRAIN_DESC  = "maximum number of train queries to process/use";
  public final static String MAX_NUM_QUERY_TRAIN_PARAM = "max_num_query_train";
  
  public final static String MAX_NUM_QUERY_TEST_DESC  = "maximum number of train queries to process/use";
  public final static String MAX_NUM_QUERY_TEST_PARAM = "max_num_query_test";
  
  public final static String DATA_FILE_DESC = "A data file";
  public final static String DATA_FILE_PARAM = "data_file";
    
  public final static String OUT_INDEX_DESC = "A directory to store index";
  public final static String OUT_INDEX_PARAM = "index_dir";
  
  public final static String MIN_PROB_PARAM  = "min_prob";
  public final static String MIN_PROB_DESC   = "A minimum probability";
      
  public final static String MAX_WORD_QTY_PARAM = "max_word_qty";
  public final static String MAX_WORD_QTY_DESC  = "A maximum number of words";

  public final static String SAVE_STAT_FILE_PARAM = "save_stat_file";
  public final static String SAVE_STAT_FILE_DESC  = "A file to save some vital query execution statistics";
  
  public final static String USE_THREAD_POOL_PARAM = "use_thread_pool";
  public final static String USE_THREAD_POOL_DESC = "Use a thread pool instead of a round-robin division of queries among threads";
  
  public final static String FIELD_NAME_PARAM = "field_name";
  public final static String FIELD_NAME_DESC = "The name of a field to process";
  
  public static final String QUERY_FIELD_NAME_PARAM = "query_field";
  public final static String QUERY_FIELD_NAME_DESC = "The name of a query field";
  
  public static final String INDEX_FIELD_NAME_PARAM = "index_field";
  public final static String INDEX_FIELD_NAME_DESC = "The name of an index field";

  public final static String FOWARD_INDEX_TYPE_PARAM = "fwd_index_type";
  public final static String FOWARD_INDEX_TYPE_DESC  = "A forward index type: " + ForwardIndex.getIndexTypeList();

  
  public final static String FOWARD_INDEX_STORE_TYPE_PARAM = "fwd_index_store_type";
  public final static String FOWARD_INDEX_STORE_TYPE_DESC  = "A forward index storage type: " + ForwardIndex.getIndexStorageTypeList();
 
  public static final String OUTPUT_FILE_PARAM = "out_file";
  public static final String OUTPUT_FILE_DESC = "Output file";
  
  public static final String BATCH_SIZE_PARAM = "batch_size";
  public static final String BATCH_SIZE_DESC = "batch size";
  
  public static final int USAGE_WIDTH = 90;
  

}
