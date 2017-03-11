#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, stackoverflow (1st arg)"
  exit 1
fi

QREL_TYPE=$2
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""

# A very moderate expansion
GIZA_EXPAND_QTY=5

# Let's always retrieve at least MAX_N top records using Lucene
MAX_N=100
if [ "$collect" = "manner" ] ; then
  NUM_RET_LIST="10,15,17,36,72,$MAX_N"
else
  NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,$MAX_N"
fi

MAX_NUM_QUERY=$3

if [ "$MAX_NUM_QUERY" = "" ] ; then
  echo "Specify the maximum number of queries to be used from the test set (2d arg)!"
  exit 1
fi

WORD_EMBEDDINGS="word2vec_retro_unweighted_minProb=0.001.txt"

EXPER_DIR_BASE="results/final/${collect}/$QREL_FILE/test/lucene/"

# 1. Warm up 
cmd="scripts/exper/test_final_model.sh  -max_num_query $MAX_NUM_QUERY ${collect} "$QREL_FILE" test lucene_giza $EXPER_DIR_BASE/giza_expand/exper@bm25=text exper@bm25=text+simple_tran=text  nmslib/${collect}/models/1_0_feature.model  $NUM_RET_LIST $WORD_EMBEDDINGS -giza_expand_qty $GIZA_EXPAND_QTY -dont_delete_trec_runs"
bash -c "$cmd"
check "$cmd"

# 2. Actual testing
# BM25
cmd="scripts/exper/test_final_model.sh  -max_num_query $MAX_NUM_QUERY ${collect} "$QREL_FILE" test lucene $EXPER_DIR_BASE/exper@bm25=text exper@bm25=text  nmslib/${collect}/models/one_feature.model  $NUM_RET_LIST $WORD_EMBEDDINGS  -dont_delete_trec_runs"
bash -c "$cmd"
check "$cmd"

# BM25 (giza-expand) the weight for Model 1 will be ignored
cmd="scripts/exper/test_final_model.sh  -max_num_query $MAX_NUM_QUERY ${collect} "$QREL_FILE" test lucene_giza $EXPER_DIR_BASE/giza_expand/exper@bm25=text exper@bm25=text+simple_tran=text  nmslib/${collect}/models/1_0_feature.model  $NUM_RET_LIST $WORD_EMBEDDINGS -giza_expand_qty $GIZA_EXPAND_QTY -dont_delete_trec_runs"
bash -c "$cmd"
check "$cmd"

cmd="scripts/exper/test_final_model.sh  -max_num_query $MAX_NUM_QUERY ${collect} "$QREL_FILE" test lucene_giza $EXPER_DIR_BASE/giza_expand_wght/exper@bm25=text exper@bm25=text+simple_tran=text  nmslib/${collect}/models/1_0_feature.model  $NUM_RET_LIST $WORD_EMBEDDINGS -giza_expand_qty $GIZA_EXPAND_QTY -giza_wght_expand -dont_delete_trec_runs"
bash -c "$cmd"
check "$cmd"



# BM25 + Model 1 (perhaps for more than one field)
if [ "$collect" = "squad" ] ; then
  MODEL_TYPE="model1=text+minProbModel1=text:1e-3+model1=text_alias1+minProbModel1=text_alias1:1e-3"
else
  MODEL_TYPE="model1=text"
fi

cmd="scripts/exper/test_final_model.sh  -max_num_query $MAX_NUM_QUERY ${collect} "$QREL_FILE" test lucene $EXPER_DIR_BASE/exper@bm25=text+${MODEL_TYPE} exper@bm25=text+${MODEL_TYPE}  nmslib/${collect}/models/out_${collect}_train_exper@bm25=text+${MODEL_TYPE}_15.model  $NUM_RET_LIST $WORD_EMBEDDINGS -dont_delete_trec_runs"
bash -c "$cmd"
check "$cmd"

