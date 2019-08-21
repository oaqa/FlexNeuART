#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

FEATURE_DESC_FILE=$2
if [ "$FEATURE_DESC_FILE" = "" ] ; then
  echo "Specify a feature description file (2d arg)"
  exit 1
fi

PARALLEL_EXPER_QTY=$3
if [ "$PARALLEL_EXPER_QTY" = "" ] ; then
  echo "Specify a number of experiments that are run in parallel (3d arg)!"
  exit 1
fi

MAX_QUERY_QTY="$4"

NUM_CPU_CORES="$5"

if [ "$NUM_CPU_CORES" = "" ] ; then
  NUM_CPU_CORES=`getNumCpuCores`
fi
if [ "$NUM_CPU_CORES" = "" ] ; then
  echo "Bug: NUM_CPU_CORES is unset!"
  exit 1
fi

THREAD_QTY=$(($NUM_CPU_CORES/$PARALLEL_EXPER_QTY))

echo "The number of CPU cores:      $NUM_CPU_CORES"
echo "The number of || experiments: $PARALLEL_EXPER_QTY"
echo "The number of threads:        $THREAD_QTY"
echo "Max # of queries to use:      $MAX_QUERY_QTY"

scripts/exper/run_feature_exper_aux.sh $collect $FEATURE_DESC_FILE $PARALLEL_EXPER_QTY $THREAD_QTY $MAX_QUERY_QTY 

