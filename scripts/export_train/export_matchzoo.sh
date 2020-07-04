#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh
source scripts/export_train/export_common.sh

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
checkVarNonEmpty "partTrain"
checkVarNonEmpty "partTest"

if [ "$outSubdir" = "" ] ; then
  outSubdir="match_zoo_train"
fi

outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/outSubdir"

if [ ! -d "$outDir" ] ; then
  mkdir "$outDir"
fi

echo "Train split: $partTrain"
echo "Eval split: $partTest"
echo "Output directory: $outDir"

target/appassembler/bin/ExportTrainPairs \
-export_fmt match_zoo  \
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
\
-out_file_train $outDir/train.tsv \
-out_file_test $outDir/test.tsv


