#!/bin/bash
# A simple script to transplant validation data from one
# set of exported data, e.g., based on the original
# MSMARCO set of queries to another one, e.g., ORCAS.
# This makes training on differently exported training
# data comparable. This only makes sense when
# "transplating" runs within the *SAME* collection.
set -eo pipefail

source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

boolOpts=("h" "help" "print help")

usageMain="<collection> \
<source sub-dir in $DERIVED_DATA_SUBDIR> <source sub-dir index field name> \
<target sub-dir in $DERIVED_DATA_SUBDIR> <target sub-dir index field name>"

parseArguments $@

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

srcSubDir=${posArgs[1]}
if [ "$srcSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify source sub-dir in $DERIVED_DATA_SUBDIR (2d arg)"
  exit 1
fi

srcIndexFieldName=${posArgs[2]}
if [ "$srcIndexFieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the name of the *SOURCE* index field (3rd arg)"
  exit 1
fi

dstSubDir=${posArgs[3]}
if [ "dstSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify target sub-dir in $DERIVED_DATA_SUBDIR (4th arg)"
  exit 1
fi

dstIndexFieldName=${posArgs[4]}
if [ "$dstIndexFieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the name of the *SOURCE* index field (5th arg)"
  exit 1
fi

srcDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$srcSubDir/$srcIndexFieldName"
dstDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$dstSubDir/$dstIndexFieldName"


echo "======================================================="
echo "Source directory: $srcDir"
echo "Target directory: $dstDir"
echo "======================================================="

set -x

if [ "$srcDir" = "$dstDir" ] ; then
  echo "Source and target directories should be different"
  exit 1
fi

if [ ! -d "$srcDir" ] ; then
  echo "Not a directory: $srcDir"
  exit 1
fi

if [ ! -d "$dstDir" ] ; then
  echo "Not a directory: $dstDir"
  exit 1
fi

cd "$srcDir"
cut -f 3 test_run.txt |sort -u > did_list.txt
cut -f 1 test_run.txt |sort -u > qid_list.txt

fgrep data_docs.tsv -f did_list.txt > data_docs_val.tsv
fgrep data_query.tsv -f qid_list.txt > data_query_val.tsv

srcDirFullPath=`pwd`
cd -
cd "$dstDir"
# It's ok if this creates a few duplicates, they will be
# ignored when a train.py reads input data.
cat $srcDirFullPath/data_docs_val.tsv >> data_docs.tsv
cat $srcDirFullPath/data_query_val.tsv >> data_query.tsv

mv test_run.txt test_run_orig_exported.txt
cp qrels.txt qrels_run_orig_exported.txt

cp $srcDirFullPath/test_run.txt .
cat $srcDirFullPath/qrels.txt >> qrels.txt


echo "Finished successfully!"
