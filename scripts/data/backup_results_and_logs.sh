#!/bin/bash
. scripts/common.sh

if [ "$KNN4QA_COLLECT" = "" ] ; then
  echo "Error environment variable KNN4QA_COLLECT is not set, should be compr,stackoverflow,squad, ..."
  exit 1
fi

cp logs.* ~/data/logs/$KNN4QA_COLLECT
check "cp logs.* ~/data/logs/$KNN4QA_COLLECT"

cp server.logs.* ~/data/logs/$KNN4QA_COLLECT
check "cp server.logs.* ~/data/logs/$KNN4QA_COLLECT"

dt=`date +%Y-%m-%d` 
check "dt ..."

tar cvfj results_${KNN4QA_COLLECT}_${dt}.bz2 results*
check "tar cvfj results_${KNN4QA_COLLECT}_${dt}.bz2 results*"

tar cvfj logs_${KNN4QA_COLLECT}_${dt}.bz2 ~/data/logs/$KNN4QA_COLLECT/*"
check "tar cvfj logs_${KNN4QA_COLLECT}_${dt}.bz2 ~/data/logs/$KNN4QA_COLLECT*"
