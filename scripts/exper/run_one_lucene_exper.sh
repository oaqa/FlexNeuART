#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi


# These flags are mostly for debug purposes
regen_feat="1"
recomp_model="1"
rerun_lucene="1"
test_model_results="0"

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

N_TRAIN=$8

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

checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "TRAIN_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "LUCENE_CACHE_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FAKE_RUN_ID"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
gizaRootDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi


cacheDir="$COLLECT_ROOT/$collect/$LUCENE_CACHE_SUBDIR/"

if [ ! -d "$cacheDir" ] ; then
  mkdir -p "$cacheDir"
  check "mkdir -p $cacheDir"
fi

for part in "$TRAIN_SUBDIR" "$TEST_PART" ; do
  if [ ! -d "$cacheDir/$part" ] ; then
   mkdir -p "$cacheDir/$part"
   check "mkdir -p $cacheDir/$part"
  fi
done

if [ "$maxQueryQtyTrain" = "" ] ; then
  CACHE_FILE_TRAIN="$cacheDir/$TRAIN_SUBDIR/all_queries"
else
  CACHE_FILE_TRAIN="$cacheDir/$TRAIN_SUBDIR/max_query_qty=$maxQueryQtyTrain"
fi

if [ "$maxQueryQtyTest" = "" ] ; then
  CACHE_FILE_TEST="$cacheDir/$TEST_PART/all_queries"
else
  CACHE_FILE_TEST="$cacheDir/$TEST_PART/max_query_qty=$maxQueryQtyTrain"
fi

URI="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR"

echo "==============================================="
echo "Using $TEST_PART for testing!"
echo "Experiment directory:           $EXPER_DIR"
echo "QREL file:                      $QREL_FILE"
echo "Directory with TREC-style runs: $TREC_RUN_DIR"
echo "Report directory:               $REPORT_DIR"
echo "Query cache file (for training):$CACHE_FILE_TRAIN"
echo "Query cache file (for testing): $CACHE_FILE_TEST"
echo "==============================================="
echo "Data directory:          $inputDataDir"
echo "Data file name:          $queryFileName"
echo "Forward index directory: $fwdIndexDir"
echo "Lucene index directory:  $URI"
echo "Embedding directory:     $embedDir"
echo "GIZA root directory:     $gizaRootDir"



OUT_PREF_TRAIN="out_${collect}_${TRAIN_SUBDIR}"
OUT_PREF_TEST="out_${collect}_${TEST_PART}"
FULL_OUT_PREF_TRAIN="$EXPER_DIR/$OUT_PREF_TRAIN"
FULL_OUT_PREF_TEST="$EXPER_DIR/$OUT_PREF_TEST"

query_log_file=${TREC_RUN_DIR}/query.log
check "query_log_file=${TREC_RUN_DIR}/query.log"


resourceDirParams=" -fwd_index_dir \"$fwdIndexDir\" -embed_dir \"$embedDir\" -giza_root_dir \"$gizaRootDir\" -giza_iter_qty $GIZA_ITER_QTY "

if [ "$EXTR_TYPE" != "none" ] ; then
  if [ "$N_TRAIN" = "" ] ; then
    echo "Specify the numbers of candidate records retrieved for the training subset for each query (8th arg)!"
    exit 1
  fi


  if [ "$regen_feat" = "1" ] ; then
    scripts/query/gen_features.sh $collect \
                                  $QREL_FILE \
                                  "$TRAIN_SUBDIR" \
                                  lucene $URI $N_TRAIN \
                                  "$EXTR_TYPE" "$EXPER_DIR" $maxQueryQtyTrainParam  \
                                  -out_pref "$OUT_PREF_TRAIN" \
                                  -thread_qty $THREAD_QTY \
                                  -query_cache_file $CACHE_FILE_TRAIN 2>&1
    check "gen_features.sh $collect ... "
  fi

  MODEL_FILE="${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.model"

  if [ "$recomp_model" = "1" ] ; then
    model_log_file="$EXPER_DIR/model.log"
    echo > $model_log_file
    
    scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART $METRIC_TYPE 2>&1 | tee -a "$model_log_file"
    check_pipe "scripts/letor/ranklib_train_coordasc.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $NUM_RAND_RESTART $METRIC_TYPE 2>&1 "

    if [ "$test_model_results" = "1" ] ; then
      scripts/query/run_query.sh  -u "$URI" -cand_prov lucene -q "$inputDataDir/$TRAIN_SUBDIR/$queryFileName" -n "$N_TRAIN" \
                                  -run_id $FAKE_RUN_ID \
                                  -o $TREC_RUN_DIR/run_check_train_metrics  -thread_qty $THREAD_QTY  \
                                  $resourceDirParams \
                                  -extr_type_final "$EXTR_TYPE" -model_final "$MODEL_FILE" $maxQueryQtyTrainParam \
                                  -query_cache_file $CACHE_FILE_TRAIN 2>&1
      check "run_query.sh"

      scripts/exper/eval_output.py "$inputDataDir/$TRAIN_SUBDIR/$QREL_FILE" "${TREC_RUN_DIR}/run_check_train_metrics_${N_TRAIN}"
      check "eval_output.py"

      echo "Model tested, now exiting!"
      exit 0
    fi
  fi

  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -cand_prov lucene -q "$inputDataDir/$TEST_PART/$queryFileName"  -n "$NTEST_STR" \
                                -run_id $FAKE_RUN_ID \
                                -o $TREC_RUN_DIR/run   -thread_qty $THREAD_QTY \
                                $resourceDirParams \
                                -extr_type_final "$EXTR_TYPE" -model_final "$MODEL_FILE" $maxQueryQtyTestParam \
                                -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
else
  if [ "$rerun_lucene" = 1 ] ; then
    scripts/query/run_query.sh  -u "$URI" -cand_prov lucene -q "$inputDataDir/$TEST_PART/$queryFileName"  -n "$NTEST_STR" \
                                -run_id $FAKE_RUN_ID \
                                -o $TREC_RUN_DIR/run -thread_qty $THREAD_QTY $maxQueryQtyTestParam \
                                -query_cache_file $CACHE_FILE_TEST 2>&1|tee $query_log_file
    check_pipe "run_query.sh"
  fi
fi

. scripts/exper/common_eval.sh
