#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

QREL_TYPE=$2
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""

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

