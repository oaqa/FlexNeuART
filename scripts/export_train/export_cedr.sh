#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh
source scripts/export_train/export_common.sh

CEDR_DOCS_FILE=data_docs.tsv
CEDR_QUERY_FILE=data_query.tsv
CEDR_TEST_RUN_FILE=test_run.txt
CEDR_TRAIN_PAIRS_FILE=train_pairs.tsv

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QUERY_FIELD_NAME"
checkVarNonEmpty "QREL_FILE"

checkVarNonEmpty "CEDR_DOCS_FILE"
checkVarNonEmpty "CEDR_QUERY_FILE"
checkVarNonEmpty "CEDR_TEST_RUN_FILE"
checkVarNonEmpty "CEDR_TRAIN_PAIRS_FILE"

checkVarNonEmpty "indexFieldName"
checkVarNonEmpty "threadQty"
checkVarNonEmpty "candTrainQty"
checkVarNonEmpty "candTestQty"
checkVarNonEmpty "partTrain"
checkVarNonEmpty "partTest"

if [ "$outSubdir" = "" ] ; then
  outSubdir="cedr_train"
fi

outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/outSubdir/$indexFieldName"

if [ ! -d "$outDir" ] ; then
  mkdir -p "$outDir"
fi

echo "Train split: $partTrain"
echo "Eval split: $partTest"
echo "Output directory: outDir"

cat "$inputDataDir/$partTrain/$QREL_FILE" "$inputDataDir/$partTest/$QREL_FILE"  > "$outDir/$QREL_FILE"

setJavaMem 1 2

target/appassembler/bin/ExportTrainPairs \
-export_fmt cedr  \
-cand_train_qty $candTrainQty -cand_test_qty $candTestQty \
$maxNumQueryTestParam \
$maxNumQueryTrainParam \
\
-fwd_index_dir  "$fwdIndexDir" \
\
-u "$luceneIndexDir" \
-thread_qty $threadQty \
-sample_neg_qty $sampleNegQty \
\
-index_field $indexFieldName \
-query_field $QUERY_FIELD_NAME \
\
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
\
-data_file_docs "$outDir/$CEDR_DOCS_FILE" \
-data_file_queries "$outDir/$CEDR_QUERY_FILE" \
-test_run_file "$outDir/$CEDR_TEST_RUN_FILE" \
-train_pairs_file "$outDir/$CEDR_TRAIN_PAIRS_FILE" 



