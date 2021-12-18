#!/bin/bash -e
# Set this var before common scripts
outSubdir="cedr_train"

source ./common_proc.sh
source ./config.sh
source ./export_train/export_common.sh


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "QUESTION_FILE_PREFIX"

CEDR_DOCS_FILE="data_docs.tsv"
CEDR_QUERY_FILE="data_query.tsv"
CEDR_TEST_RUN_FILE="test_run.txt"
CEDR_TRAIN_PAIRS_FILE="train_pairs.tsv"

checkVarNonEmpty "randSeed"
checkVarNonEmpty "indexExportFieldName"
checkVarNonEmpty "queryExportFieldName"
checkVarNonEmpty "threadQty"
checkVarNonEmpty "candTrainQty"
checkVarNonEmpty "candTestQty"
checkVarNonEmpty "partTrain"
checkVarNonEmpty "partTest"

checkVarNonEmpty "hardNegQty"
checkVarNonEmpty "sampleMedNegQty"
checkVarNonEmpty "sampleEasyNegQty"

# Sometimes QREL files come without a trailing EOL
# So we want to add it 
outFile="$outDir/$QREL_FILE"
echo -n "" > $outFile
for partCurr in $partTrain $partTest ; do 
  cat "$inputDataDir/$partCurr/$QREL_FILE" >> $outFile
  echo "" >> $outFile
done

NO_MAX=1
setJavaMem 1 8 $NO_MAX

ExportTrainPairs \
$handleCaseParam \
-seed $randSeed \
-export_fmt cedr  \
-cand_train_qty $candTrainQty -cand_test_qty $candTestQty \
$maxNumQueryTestParam \
$maxNumQueryTrainParam \
-max_doc_whitespace_qty $maxDocWhitespaceQty \
\
$commonResourceParams \
\
$candProvParams \
-thread_qty $threadQty \
\
-hard_neg_qty $hardNegQty \
-sample_med_neg_qty $sampleMedNegQty \
-sample_easy_neg_qty $sampleEasyNegQty \
\
-index_export_field $indexExportFieldName \
-query_export_field $queryExportFieldName \
\
-query_file_train_pref "$inputDataDir/$partTrain/$QUESTION_FILE_PREFIX" \
-qrel_file_train "$inputDataDir/$partTrain/$QREL_FILE" \
-query_file_test_pref "$inputDataDir/$partTest/$QUESTION_FILE_PREFIX" \
-qrel_file_test "$inputDataDir/$partTest/$QREL_FILE" \
\
-data_file_docs "$outDir/$CEDR_DOCS_FILE" \
-data_file_queries "$outDir/$CEDR_QUERY_FILE" \
-test_run_file "$outDir/$CEDR_TEST_RUN_FILE" \
-train_pairs_file "$outDir/$CEDR_TRAIN_PAIRS_FILE" 



