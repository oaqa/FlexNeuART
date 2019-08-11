#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# Quite a few things in this script are still hard-coded, e.g. these values:
INDEX_FIELD_NAME=text
QUERY_FIELD_NAME=text
CAND_QTY=100
THREAD_QTY=4

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
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
  exit 1
fi

part=dev1
scripts/data/run_export_train_text_pairs.sh -cand_qty $CAND_QTY -export_fmt match_zoo  -max_num_query 5000 -fwd_index_dir  "$fwdIndexDir" -thread_qty $THREAD_QTY -sample_neg_qty -1 -index_field $INDEX_FIELD_NAME -q $inputDataDir/$part/QuestionFields.jsonl -u "$luceneIndexDir" -qrel_file $inputDataDir/$part/$QREL_FILE -query_field $QUERY_FIELD_NAME -out_file $matchZooTrainDir/${part}_allCand.tsv

part=train_bitext
scripts/data/run_export_train_text_pairs.sh -cand_qty $CAND_QTY -export_fmt match_zoo  -max_num_query 50000 -fwd_index_dir  "$fwdIndexDir" -thread_qty $THREAD_QTY -sample_neg_qty 10 -index_field $INDEX_FIELD_NAME -q $inputDataDir/$part/QuestionFields.jsonl -u "$luceneIndexDir" -qrel_file $inputDataDir/$part/$QREL_FILE -query_field $QUERY_FIELD_NAME -out_file $matchZooTrainDir/${part}_neg10.tsv

