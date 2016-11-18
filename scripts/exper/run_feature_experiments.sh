#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

QREL_TYPE=$2
QREL_FILE=""
if [ "$QREL_TYPE" = "graded" ] ; then
  QREL_FILE="qrels_all_graded.txt"
elif [ "$QREL_TYPE" = "binary" ] ; then
  QREL_FILE="qrels_all_binary.txt"
elif [ "$QREL_TYPE" = "onlybest" ] ; then
  QREL_FILE="qrels_onlybest.txt"
elif [ "$QREL_TYPE" = "graded_same_score" ] ; then
  QREL_FILE="qrels_all_graded_same_score.txt"
elif [ "$QREL_TYPE" = "" ] ; then
  echo "Specifiy QREL type (2rd arg)"
  exit 1
else
  echo "Unsupported QREL type (2rd arg) $QREL_TYPE, expected binary, onlybest, graded, graded_same_score"
  exit 1
fi

if [ "$QREL_FILE" = "" ] ; then
  echo "Bug: QREL_FILE is empty for some reason!"
  exit 1
fi

FEATURE_DESC_FILE="$3"
if [ "$FEATURE_DESC_FILE" = "" ] ; then
  echo "Specify a feature description file (3d arg)"
  exit 1
fi
if [ ! -f "$FEATURE_DESC_FILE" ] ; then
  echo "Not a file (2d arg)"
  exit 1
fi

PARALLEL_EXPER_QTY=$4
if [ "$PARALLEL_EXPER_QTY" = "" ] ; then
  echo "Specify a number of experiments that are run in parallel (4th arg)!"
  exit 1
fi

MAX_QUERY_QTY="$5"

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

NUM_CPU_CORES=$6

if [ "$NUM_CPU_CORES" = "" ] ; then
  NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
  check "getting the number of CPU cores, do you have /proc/cpu/info?"
fi

THREAD_QTY=$(($NUM_CPU_CORES/$PARALLEL_EXPER_QTY))

echo "The number of CPU cores:      $NUM_CPU_CORES"
echo "The number of || experiments: $PARALLEL_EXPER_QTY"
echo "The number of threads:        $THREAD_QTY"
echo "Max # of queries to use:      $MAX_QUERY_QTY"
echo "QREL file:                    $QREL_FILE"

EXPER_DIR="results/feature_exper/"

scripts/exper/run_feature_exper_aux.sh $collect $QREL_FILE $EXPER_DIR $FEATURE_DESC_FILE $PARALLEL_EXPER_QTY $THREAD_QTY $MAX_QUERY_QTY 

