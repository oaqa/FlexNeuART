#!/bin/bash
. scripts/common.sh
. scripts/common_nmslib.sh

#collect="compr"
#echo "Collection: $collect"
collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

TEST_PART=${POS_ARGS[1]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (2d arg)"
  exit 1 
fi

QREL_TYPE=${POS_ARGS[2]}
QREL_FILE=`get_qrel_file $QREL_TYPE "3d"`
check ""

. scripts/exper/params/swgraph_exper1_bm25_symm_text.sh
. scripts/exper/test_nmslib_server_index_common.sh

