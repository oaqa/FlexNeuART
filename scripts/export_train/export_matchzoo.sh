#!/bin/bash -e
# Set this var before common scripts
outSubdir="match_zoo_train"

source scripts/common_proc.sh
source scripts/config.sh
source scripts/export_train/export_common.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QUERY_FIELD_NAME"
checkVarNonEmpty "QREL_FILE"

checkVarNonEmpty "randSeed"
checkVarNonEmpty "indexFieldName"
checkVarNonEmpty "threadQty"
checkVarNonEmpty "candTrainQty"
checkVarNonEmpty "candTestQty"
checkVarNonEmpty "partTrain"
checkVarNonEmpty "partTest"

checkVarNonEmpty "hardNegQty"
checkVarNonEmpty "sampleMedNegQty"
checkVarNonEmpty "sampleEasyNegQty"

target/appassembler/bin/ExportTrainPairs \
-seed $randSeed \
-export_fmt match_zoo  \
\
-cand_train_qty $candTrain -cand_test_qty $candTestQty \
$maxNumQueryTestParam \
$maxNumQueryTrainParam \
-max_doc_whitespace_qty $maxDocWhitespaceQty \
\
-thread_qty $threadQty \
\
-index_field $indexFieldName \
-query_field $queryFieldName \
\
-query_field $QUERY_FIELD_NAME \
\
$commonResourceParams \
\
$candProvParams \
\
-hard_neg_qty $hardNegQty \
-sample_med_neg_qty $sampleMedNegQty \
-sample_easy_neg_qty $sampleEasyNegQty \
\
-query_file_train "$inputDataDir/$partTrain/QuestionFields.jsonl" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test "$inputDataDir/$partTest/QuestionFields.jsonl" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
\
-out_file_train $outDir/train.tsv \
-out_file_test $outDir/test.tsv


