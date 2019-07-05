#!/bin/bash -e
. scripts/common.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

fieldList=$2
if [ "$fieldList" = "" ] ; then
  echo "Specify a list of fields to index (2d arg), e.g., 'text text_unlemm'"
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
inmem_text_indx=$4
inmem_text_indx_param=""
if [ "$inmem_text_indx" = "" ] ; then
  echo "Specify in-memory (text-only) index flag (4th arg): 1 or 0"
  exit 1
fi
if [ "$inmem_text_indx" = "1" ] ; then
  inmem_text_indx_param=" -inmem_index "
fi



indexDir="memfwdindex/$collect/"
dataDir="output/$collect/"

echo "=========================================================================="
echo "Data directory: $dataDir"
echo "Index directory: $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previous index (if exists)"
  rm -rf "$indexDir"/*
fi

echo "Storing word id seq param: $store_word_id_seq_param"
echo "Using text index param:    $inmem_text_indx_param"
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

for field in $fieldList ; do
  scripts/index/run_inmemfwd_index.sh $inmem_text_indx_param $store_word_id_seq_param -root_dir "$dataDir"  -index_dir "$indexDir" -sub_dirs "$dirList" -data_file "$currFile" -field "$field"
done

