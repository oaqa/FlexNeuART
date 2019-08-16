#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# Quite a few things in this script are still hard-coded, e.g. these values:
INDEX_FIELD_NAME=text_raw
QUERY_FIELD_NAME=text

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
trainDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/cedr"

if [ ! -d "$trainDir" ] ; then
  mkdir "$trainDir"
fi

partTest=dev1
partTrain=train_bitext
sampleNegQty=20
candQty=50
maxNumQueryTest=3000
maxNumQueryTrain=1000000

scripts/data/run_export_train_text_pairs.sh -cand_qty $candQty -export_fmt cedr  \
-max_num_query_test $maxNumQueryTest -max_num_query_train $maxNumQueryTrain \
-fwd_index_dir  "$fwdIndexDir" \
-data_file_docs "$trainDir/data_docs.tsv" \
-data_file_queries "$trainDir/data_query.tsv" \
-test_run_file "$trainDir/test_run.txt" \
-train_pairs_file "$trainDir/train_pairs.tsv" \
-u "$luceneIndexDir" \
-thread_qty $threadQty \
-sample_neg_qty $sampleNegQty \
-index_field $INDEX_FIELD_NAME \
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
-query_field $QUERY_FIELD_NAME 

