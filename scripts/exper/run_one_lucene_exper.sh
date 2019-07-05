#!/bin/bash
. scripts/config.sh
. scripts/common.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr2M, compr"
  exit 1
fi


# These flags are mostly for debug purposes
regen_feat="1"
recomp_model="1"
rerun_lucene="1"
test_model_results="0"

EMBED_ROOT_DIR="WordEmbeddings"


if [ "$collect" = "manner" ] ; then
  train_part="train"
else
  #train_part="train1"
  train_part="train"
fi


QREL_FILE=$2
if [ "$QREL_FILE" = "" ] ; then
  echo "Specify QREL file name (2d arg)!"
  exit 1
fi

EXPER_DIR_BASE=$3
if [ "$EXPER_DIR_BASE" = "" ] ; then
  echo "Specify a working directory (3d arg)!"
  exit 1
fi

EXTR_TYPE="$4"

if [ "$EXTR_TYPE" = "" ] ; then
  echo "Specify a feature extractor type (4th arg)"
  exit 1
fi

MAX_QUERY_QTY="$5"

if [ "$MAX_QUERY_QTY" != "" ] ; then
  maxQueryQtyList=(`echo $MAX_QUERY_QTY|sed 's/,/ /g'`)
  len=${#maxQueryQtyList[*]}
  if [ "$len" != "2" ] ; then
    echo "If MAX_QUERY_QTY is specified (4th arg), it must have TWO comma-separated numbers (you specified $MAX_QUERY_QTY)"
    exit 1
  fi
  maxQueryQtyTrain=${maxQueryQtyList[0]}
  maxQueryQtyTest=${maxQueryQtyList[1]}
  maxQueryQtyTrainParam=" -max_num_query ${maxQueryQtyList[0]}"
  maxQueryQtyTestParam=" -max_num_query ${maxQueryQtyList[1]}"
fi

TEST_PART="$6"

if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (6th arg)"
  exit 1 
fi

THREAD_QTY=$7
if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify a number of threads for the feature extractor (7th arg)!"
  exit 1
fi

NTEST_STR=$8
if [ "$NTEST_STR" = "" ] ; then
  echo "Specify a comma-separated list of candidate record # retrieved for testing for each query (8th arg)!"
  exit 1
fi

N_TRAIN=$9

NTEST_LIST=`echo $NTEST_STR|sed 's/,/ /g'`

EXPER_DIR="$EXPER_DIR_BASE/exper"
TREC_RUN_DIR="$EXPER_DIR_BASE/trec_runs"
REPORT_DIR="$EXPER_DIR_BASE/rep"

mkdir -p "$EXPER_DIR"
check "mkdir -p $EXPER_DIR"
mkdir -p "$TREC_RUN_DIR"
check "mkdir -p $TREC_RUN_DIR"
mkdir -p "$REPORT_DIR"
check "mkdir -p $REPORT_DIR"

echo "Deleting old reports from the directory: ${REPORT_DIR}"
rm -f ${REPORT_DIR}/*
check "rm -f ${REPORT_DIR}/*"

CACHE_DIR="cache/lucene/$collect/"

if [ ! -d "$CACHE_DIR" ] ; then
  mkdir -p "$CACHE_DIR"
  check "mkdir -p $CACHE_DIR"
fi

if [ ! -d "$CACHE_DIR/$train_part" ] ; then
  mkdir -p "$CACHE_DIR/$train_part"
  check "mkdir -p $CACHE_DIR/$train_part"
fi

if [ ! -d "$CACHE_DIR/$TEST_PART" ] ; then
  mkdir -p "$CACHE_DIR/$TEST_PART"
  check "mkdir -p $CACHE_DIR/$TEST_PART"
fi

if [ "$maxQueryQtyTrain" = "" ] ; then
  CACHE_FILE_TRAIN="$CACHE_DIR/$train_part/all_queries"
else
  CACHE_FILE_TRAIN="$CACHE_DIR/$train_part/max_query_qty=$maxQueryQtyTrain"
fi

if [ "$maxQueryQtyTest" = "" ] ; then
  CACHE_FILE_TEST="$CACHE_DIR/$TEST_PART/all_queries"
else
  CACHE_FILE_TEST="$CACHE_DIR/$TEST_PART/max_query_qty=$maxQueryQtyTrain"
fi

echo "Using $TEST_PART for testing!"
echo "Experiment directory:           $EXPER_DIR"
echo "QREL file:                      $QREL_FILE"
echo "Directory with TREC-style runs: $TREC_RUN_DIR"
echo "Report directory:               $REPORT_DIR"
echo "Query cache file (for training):$CACHE_FILE_TRAIN"
echo "Query cache file (for testing): $CACHE_FILE_TEST"

URI="lucene_index/$collect"


OUT_PREF_TRAIN="out_${collect}_${train_part}"
OUT_PREF_TEST="out_${collect}_${TEST_PART}"
FULL_OUT_PREF_TRAIN="$EXPER_DIR/$OUT_PREF_TRAIN"
FULL_OUT_PREF_TEST="$EXPER_DIR/$OUT_PREF_TEST"

query_log_file=${TREC_RUN_DIR}/query.log
check "query_log_file=${TREC_RUN_DIR}/query.log"

if [ "$EXTR_TYPE" != "none" ] ; then
  if [ "$N_TRAIN" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (8th arg)!"
    exit 1
  fi


  if [ "$regen_feat" = "1" ] ; then
    scripts/query/gen_features.sh $collect $QREL_FILE $train_part lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $maxQueryQtyTrainParam  -out_pref "$OUT_PREF_TRAIN" -thread_qty $THREAD_QTY -query_cache_file $CACHE_FILE_TRAIN 2>&1
    check "scripts/query/gen_features.sh $collect $QREL_FILE $train_part lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $maxQueryQtyTrainParam  -out_pref "$OUT_PREF_TRAIN" -thread_qty $THREAD_QTY -query_cache_file $CACHE_FILE_TRAIN 2>&1"
  fi

  MODEL_FILE="${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.model"

  if [ "$recomp_model" = "1" ] ; then
    model_log_file="$EXPER_DIR/model.log"
    echo > $model_log_file
    
    scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART $METRIC_TYPE 2>&1 | tee -a "$model_log_file"
    check_pipe "scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART $METRIC_TYPE 2>&1 "

    if [ "$test_model_results" = "1" ] ; then
      scripts/query/run_query.sh  -u "$URI" -q output/$collect/${train_part}/SolrQuestionFile.txt  -n "$N_TRAIN" -o $TREC_RUN_DIR/run_check_train_metrics  -giza_root_dir tran/$collect/ -giza_iter_qty 5 -embed_dir $EMBED_ROOT_DIR/$collect  -cand_prov lucene -memindex_dir memfwdindex/$collect -extr_type_final "$EXTR_TYPE" -thread_qty $THREAD_QTY -model_final "$MODEL_FILE" $maxQueryQtyTrainParam -query_cache_file $CACHE_FILE_TRAIN 2>&1
      check "run_query.sh"

      scripts/exper/eval_output.py "output/$collect/${train_part}/$QREL_FILE" "${TREC_RUN_DIR}/run_check_train_metrics_${N_TRAIN}"
      check "eval_output.py"

      echo "Model tested, now exiting!"
      exit 0
    fi
  fi

  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -giza_root_dir tran/$collect/ -giza_iter_qty 5 -embed_dir $EMBED_ROOT_DIR/$collect  -cand_prov lucene -memindex_dir memfwdindex/$collect -extr_type_final "$EXTR_TYPE" -thread_qty $THREAD_QTY -model_final "$MODEL_FILE" $maxQueryQtyTestParam -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
else
  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -cand_prov lucene -thread_qty $THREAD_QTY $maxQueryQtyTestParam -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
fi

. scripts/exper/common_eval.sh
