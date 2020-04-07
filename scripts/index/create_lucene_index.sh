#!/bin/bash -e
# A script to create a Lucene index
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR"

echo "=========================================================================="
echo "Data directory: $inputDataDir"
echo "Index directory: $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previously created index (if exists)"
  rm -rf "$indexDir"/*
fi
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
else
  echo "Using the data input file: $dataFileName"
fi
target/appassembler/bin/LuceneIndexer -input_data_dir "$inputDataDir" -index_dir "$indexDir" -data_sub_dirs "$dirList" -data_file "$dataFileName"
