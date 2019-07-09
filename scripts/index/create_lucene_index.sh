#!/bin/bash -e
# A script to create a Lucene index
source scripts/common_proc.sh
source scripts/common_vars.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

checkVarNonEmpty "LUCENE_INDEX_DIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$collect/$INPUT_DATA_SUBDIR"
indexDir="$collect/$LUCENE_INDEX_DIR"

echo "=========================================================================="
echo "Data directory: $inputDataDir"
echo "Index directory: $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previous index (if exists)"
  rm -rf "$indexDir"/*
fi
echo "=========================================================================="
retVal=""
getIndexDataInfo "$inputDataDir"
dirList=${retVal[0]}
dataFileName=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
else
  echo "Using the data input file: $dataFileName"
fi
scripts/index/run_lucene_index.sh -input_data_dir "$inputDataDir" -index_dir "$indexDir" -data_sub_dirs "$dirList" -data_file "$dataFileName"
