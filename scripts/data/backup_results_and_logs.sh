#!/bin/bash
. scripts/common.sh

if [ "$KNN4QA_COLLECT" = "" ] ; then
  echo "Error environment variable KNN4QA_COLLECT is not set, should be compr,stackoverflow,squad, ..."
  exit 1
fi

DATA_DIR=$1

if [ "$DATA_DIR" = "" ] ; then
  echo "Enter the name of the data directory!"
  exit 1
fi
if [ ! -d "$DATA_DIR" ] ; then
  echo "$DATA_DIR is not a directory!"
  exit 1
fi

cp log.* $DATA_DIR/logs/$KNN4QA_COLLECT
check "cp log.* $DATA_DIR/logs/$KNN4QA_COLLECT"

cp server.log.* $DATA_DIR/logs/$KNN4QA_COLLECT
check "cp server.logs.* $DATA_DIR/logs/$KNN4QA_COLLECT"

dt=`date +%Y-%m-%d` 
check "dt ..."

tar cvfj --exclude=results_* results_${KNN4QA_COLLECT}_${dt}.bz2 results*
check "tar cvfj results_${KNN4QA_COLLECT}_${dt}.bz2 results*"

tar cvfj logs_${KNN4QA_COLLECT}_${dt}.bz2 $DATA_DIR/logs/$KNN4QA_COLLECT/*
check "tar cvfj logs_${KNN4QA_COLLECT}_${dt}.bz2 $DATA_DIR/logs/$KNN4QA_COLLECT*"

tar cvfj gs_cache_${KNN4QA_COLLECT}_${dt}.bz2 gs_cache/*
check "tar cvfj gs_cache_${KNN4QA_COLLECT}_${dt}.bz2 gs_cache/*"
