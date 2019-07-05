#!/bin/bash
. scripts/common.sh
. scripts/common_nmslib.sh

collect="compr"
echo "Collection: $collect"
#collect=${POS_ARGS[0]}
#if [ "$collect" = "" ] ; then
  #echo "Specify a collection: manner, compr (1st arg)"
  #exit 1
#fi

TEST_PART=${POS_ARGS[0]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (1st arg)"
  exit 1 
fi

QREL_TYPE=${POS_ARGS[1]}
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""


. scripts/exper/params/napp_bm25_text_compr.sh
. scripts/exper/test_nmslib_server_index_common.sh

