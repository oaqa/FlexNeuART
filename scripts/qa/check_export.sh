#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# It's best to use with a our SQUAD test collection

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection, e.g, squad (1st arg)"
  exit 1
fi

testPart=$2
if [ "$testPart" = "" ] ; then
  echo "Specify a test part, e.g, ev1 (2d arg)"
  exit 1
fi

testPart=$2

MAX_NUM_DOC=50

MAX_NUM_DOC=50
MAX_NUM_QUERY=100
EPS_DIFF=1e-5

cdir=scripts/qa

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"
gizaRootDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR"

resourceDirParams=" -fwd_index_dir \"$fwdIndexDir\" -embed_dir \"$embedDir\" -giza_root_dir \"$gizaRootDir\" "

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

queryFileParam=" -q \"$inputDataDir/$testPart/$queryFileName\" "

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text_unpruned.json -model_file $cdir/bm25=text+model1=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -eps_diff $EPS_DIFF  $resourceDirParams $queryFileParam

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY   -eps_diff $EPS_DIFF  $resourceDirParams $queryFileParam

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text.json -model_file $cdir/bm25=text+model1=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -eps_diff $EPS_DIFF  $resourceDirParams $queryFileParam

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -eps_diff $EPS_DIFF  $resourceDirParams $queryFileParam

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -model_file $cdir/bm25=text+model1=text+embed=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -eps_diff $EPS_DIFF  $resourceDirParams $queryFileParam

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -eps_diff $EPS_DIFF $resourceDirParams $queryFileParam

echo "ALL CHECKS ARE DONE!"
