#!/bin/bash
. scripts/config.sh
. scripts/common.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr2M, compr"
  exit 1
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

GALAGO_OP="$4"

if [ "$GALAGO_OP" = "" ] ; then
  echo "Specify a Galago operator, e.g., combine (4th arg)"
  exit 1
fi

GALAGO_PARAMS="$5"

if [ "$GALAGO_PARAMS" = "" ] ; then
  echo "Specify a Galago parameters, or @ to indicate empty list (5th arg)"
  exit 1
fi

MAX_QUERY_QTY="$6"

if [ "$MAX_QUERY_QTY" != "" ] ; then
  maxQueryQtyParam=" -max_num_query $MAX_QUERY_QTY"
fi

TEST_PART="$7"

if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (7th arg)"
  exit 1 
fi

THREAD_QTY=$8
if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify a number of threads for the feature extractor (8th arg)!"
  exit 1
fi

NTEST_STR=$9
if [ "$NTEST_STR" = "" ] ; then
  echo "Specify a comma-separated list of candidate record # retrieved for testing for each query (9th arg)!"
  exit 1
fi

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

echo "Using $TEST_PART for testing!"
echo "Experiment directory:           $EXPER_DIR"
echo "QREL file:                      $QREL_FILE"
echo "Directory with TREC-style runs: $TREC_RUN_DIR"
echo "Report directory:               $REPORT_DIR"

URI="galago_index/$collect"


OUT_PREF_TRAIN="out_${collect}_${train_part}_${EXTR_TYPE}"
OUT_PREF_TEST="out_${collect}_${TEST_PART}_${EXTR_TYPE}"
FULL_OUT_PREF_TRAIN="$EXPER_DIR/$OUT_PREF_TRAIN"
FULL_OUT_PREF_TEST="$EXPER_DIR/$OUT_PREF_TEST"

query_log_file=${TREC_RUN_DIR}/query.log
check "query_log_file=${TREC_RUN_DIR}/query.log"

if [ "$GALAGO_PARAMS" = "@" ] ; then
  GALAGO_OPTS=""
else
  GALAGO_OPTS="-galago_params $GALAGO_PARAMS"
fi

scripts/query/run_query.sh  -u "$URI" -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -cand_prov galago -thread_qty $THREAD_QTY -galago_op $GALAGO_OP "$GALAGO_OPTS" $maxQueryQtyParam 2>&1|tee $query_log_file
check_pipe "run_query.sh"

. scripts/exper/common_eval.sh
