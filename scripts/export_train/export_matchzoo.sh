#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# Quite a few things in this script are still hard-coded, e.g. these values:
INDEX_FIELD_NAME=text
QUERY_FIELD_NAME=text

partTest=dev1
partTrain=train_bitext
sampleNegQty=10
candTrainQty=500
candTestQty=50
maxNumQueryTest=5000
maxNumQueryTrain=50000

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi
threadQty=$2
if [ "$threadQty" = "" ] ; then
  echo "Specify # of threads (2d arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QREL_FILE"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
luceneIndexDir="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR/"
matchZooTrainDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/match_zoo_train"

if [ ! -d "$matchZooTrainDir" ] ; then
  mkdir "$matchZooTrainDir"
fi

scripts/data/run_export_train_text_pairs.sh \
-cand_train_qty $candTrain -cand_test_qty $candTestQty \
-export_fmt match_zoo  \
-max_num_query_test $maxNumQueryTest -max_num_query_train $maxNumQueryTrain \
-out_file_train $matchZooTrainDir/${partTrain}_neg10.tsv \
-out_file_test $matchZooTrainDir/${partTest}_allCand.tsv \
-fwd_index_dir  "$fwdIndexDir" \
-u "$luceneIndexDir" \
-thread_qty $threadQty \
-sample_neg_qty $sampleNegQty \
-index_field $INDEX_FIELD_NAME \
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
-query_field $QUERY_FIELD_NAME 

