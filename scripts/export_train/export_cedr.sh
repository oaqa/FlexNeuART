#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# The output directory name is hard-coded

source scripts/export_train/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QUERY_FIELD_NAME"
checkVarNonEmpty "QREL_FILE"

checkVarNonEmpty "$indexFieldName"
checkVarNonEmpty "$threadQty"
checkVarNonEmpty "$candTrainQty"
checkVarNonEmpty "$candTestQty"

outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/cedr_train"

if [ ! -d "$outDir" ] ; then
  mkdir "$outDir"
fi

scripts/data/run_export_train_text_pairs.sh \
-export_fmt cedr  \

-cand_train_qty $candTrainQty -cand_test_qty $candTestQty \

$maxNumQueryTestParam \
$maxNumQueryTrainParam \
-fwd_index_dir  "$fwdIndexDir" \

-u "$luceneIndexDir" \
-thread_qty $threadQty \
-sample_neg_qty $sampleNegQty \
-index_field $indexFieldName \
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
-query_field $QUERY_FIELD_NAME
\
-cand_train_qty $candTrain -cand_test_qty $candTestQty \
$maxNumQueryTestParam \
$maxNumQueryTrainParam \
\
-thread_qty $threadQty \
\
-index_field $indexFieldName \
-fwd_index_dir  "$fwdIndexDir" \
\
-query_field $QUERY_FIELD_NAME \
-u "$luceneIndexDir" \
\
-sample_neg_qty $sampleNegQty \
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \


