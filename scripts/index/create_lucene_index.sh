#!/bin/bash -e
source scripts/common.sh
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

dataDir="output/$collect/"
indexDir="lucene_index/$collect"

echo "=========================================================================="
echo "Data directory: $dataDir"
echo "Index directory: $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previous index (if exists)"
  rm -rf "$indexDir"/*
fi
echo "=========================================================================="
retVal=""
getIndexDataInfo "$dataDir"
dirList=${retVal[0]}
currFile=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$currFile" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi
scripts/index/run_lucene_index.sh -root_dir "$dataDir" -index_dir "$indexDir" -sub_dirs "$dirList" -data_file "$currFile"
