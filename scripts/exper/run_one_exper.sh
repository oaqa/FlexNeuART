#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr2M, compr"
  exit 1
fi

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

function check_pipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
    exit 1
  fi
}


# These flags are mostly for debug purposes
regen_feat="1"
recomp_model="1"
rerun_lucene="1"


if [ "$collect" = "manner" ] ; then
  train_part="train"
else
  #train_part="train1"
  train_part="train"
fi


CACHE_DIR="cache/lucene/$collect/$train_part"

if [ ! -d "$CACHE_DIR" ] ; then
  mkdir -p "$CACHE_DIR"
fi

EXPER_DIR_BASE=$2
if [ "$EXPER_DIR_BASE" = "" ] ; then
  echo "Specify a working directory (2d arg)!"
  exit 1
fi

EXTR_TYPE="$3"

if [ "$EXTR_TYPE" = "" ] ; then
  echo "Specify a feature extractor type (3d arg)"
  exit 1
fi

MAX_QUERY_QTY="$4"

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

TEST_PART="$5"

if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (5th arg)"
  exit 1 
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

echo "Using $TEST_PART for testing!"
echo "Experiment directory:           $EXPER_DIR"
echo "Directory with TREC-style runs: $TREC_RUN_DIR"
echo "Report directory:               $REPORT_DIR"
echo "Query cache file (for training):$CACHE_FILE"

THREAD_QTY=$6
if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify a number of threads for the feature extractor (6th arg)!"
  exit 1
fi

NTEST_STR=$7
if [ "$NTEST_STR" = "" ] ; then
  echo "Specify a comma-separated list of candidate record # retrieved for testing for each query (7th arg)!"
  exit 1
fi

NTEST_LIST=`echo $NTEST_STR|sed 's/,/ /g'`

URI="lucene_index/$collect"


OUT_PREF_TRAIN="out_${collect}_${train_part}_${EXTR_TYPE}"
OUT_PREF_TEST="out_${collect}_${TEST_PART}_${EXTR_TYPE}"
FULL_OUT_PREF_TRAIN="$EXPER_DIR/$OUT_PREF_TRAIN"
FULL_OUT_PREF_TEST="$EXPER_DIR/$OUT_PREF_TEST"

query_log_file=${TREC_RUN_DIR}/query.log
check "query_log_file=${TREC_RUN_DIR}/query.log"

if [ "$EXTR_TYPE" != "none" ] ; then
  N_TRAIN=$8
  if [ "$N_TRAIN" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (8th arg)!"
    exit 1
  fi

  EMBED_FILES="$9"

  if [ "$EMBED_FILES" = "" ] ; then
    echo "Specify a (quoted) list of embedding files (9th arg)"
    exit 1
  fi

  HORDER_FILES="tran_embed.0,tran_embed.1,tran_embed.2,tran_embed.3,tran_embed.4"

  if [ "$regen_feat" = "1" ] ; then
    scripts/query/gen_features.sh $collect $train_part lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $maxQueryQtyTrainParam  -out_pref "$OUT_PREF_TRAIN" -embed_files "$EMBED_FILES" -horder_files "$HORDER_FILES" -thread_qty $THREAD_QTY -query_cache_file $CACHE_FILE_TRAIN 2>&1
    check "scripts/query/gen_features.sh $collect $train_part lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $maxQueryQtyTrainParam  -out_pref "$OUT_PREF_TRAIN" -embed_files "$EMBED_FILES" -horder_files "$HORDER_FILES" -thread_qty $THREAD_QTY -query_cache_file $CACHE_FILE_TRAIN 2>&1"
  fi

  MODEL_FILE="${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.model"

  if [ "$recomp_model" = "1" ] ; then
    NUM_RAND_RESTART=10
    model_log_file="$EXPER_DIR/model.log"
    echo > $model_log_file
    
    scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART 2>&1 | tee -a "$model_log_file"
    check_pipe "scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART 2>&1 "
  fi

  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -giza_root_dir tran/$collect/ -giza_iter_qty 5 -embed_dir WordEmbeddings/$collect  -embed_files  "$EMBED_FILES" -cand_prov lucene -memindex_dir memfwdindex/$collect -extr_type_final "$EXTR_TYPE" -thread_qty $THREAD_QTY -horder_files "$HORDER_FILES" -model_final "$MODEL_FILE" $maxQueryQtyTestParam -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
else
  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -cand_prov lucene -thread_qty $THREAD_QTY $maxQueryQtyTestParam -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
fi

QRELS="output/$collect/${TEST_PART}/qrels.txt"


rm -f "${REPORT_DIR}/out_*"

for oneN in $NTEST_LIST ; do
  echo "======================================"
  echo "N=$oneN"
  echo "======================================"
  REPORT_PREF="${REPORT_DIR}/out_${oneN}"

  scripts/exper/eval_output.py trec_eval-9.0.4/trec_eval "$QRELS"  "${TREC_RUN_DIR}/run_${oneN}" "$REPORT_PREF" "$oneN"
  check "eval_output.py"
done

echo "Deleting trec runs from the directory: ${TREC_RUN_DIR}"
rm ${TREC_RUN_DIR}/*
# There should be at least one run, so, if rm fails, it fails because files can't be deleted
check "rm ${TREC_RUN_DIR}/*" 
echo "Bzipping trec_eval output in the directory: ${REPORT_DIR}"
bzip2 ${REPORT_DIR}/*.trec_eval
check "bzip2 "${REPORT_DIR}/*.trec_eval""
