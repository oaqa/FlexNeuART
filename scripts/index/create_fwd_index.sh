#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

fieldList=$2
if [ "$fieldList" = "" ] ; then
  echo "Specify a space-separated list of fields to index (2d arg), e.g., 'text text_unlemm'"
  exit 1
fi

store_word_id_seq=$3
store_word_id_seq_param=""
if [ "$store_word_id_seq" = "" ] ; then
  echo "Specify the flag to store word sequence (3d arg): 1 or 0"
  exit 1
fi
if [ "$store_word_id_seq" = "1" ] ; then
  store_word_id_seq_param=" -store_word_id_seq "
fi
fwd_index_type=$4
if [ "$fwd_index_type" = "" ] ; then
  echo "Specify index-type (4th arg), e.g. lucene"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"

echo "=========================================================================="
echo "Data directory: $inputDataDir"
echo "Index directory: $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previous index (if exists)"
  rm -rf "$indexDir"/*
fi

echo "Storing word id seq param: $store_word_id_seq_param"
echo "Index type:                $fwd_index_type"
echo "Field list:                $fieldList"
echo "=========================================================================="
retVal=""
getIndexQueryDataInfo "$inputDataDir"
dirList=${retVal[0]}
dataFileName=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

for field in $fieldList ; do
  scripts/index/run_buildfwd_index.sh -fwd_index_type $fwd_index_type $store_word_id_seq_param -input_data_dir "$inputDataDir"  -index_dir "$indexDir" -data_sub_dirs "$dirList" -data_file "$dataFileName" -field_name "$field"
done

