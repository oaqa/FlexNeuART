#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

field=$2

if [ "$field" = "" ] ; then
  echo "Specify a document/index: e.g., text (2d arg)"
  exit 1
fi

query_field=$3

if [ "$query_field" = "" ] ; then
  echo "Specify a query field, e.g., text (3d arg)"
  exit 1
fi

part=$4

if [ "$part" = "" ] ; then
  echo "Specify an input collection sub-directory, e.g., train (4th arg)"
  exit 1
fi

outDir=$5

if [ "$outDir" = "" ] ; then
 echo "Specify output dir (5th arg)"
 exit 1
fi

if [ ! -d "$outDir" ] ; then
  "$outDir isn't a directory"
  exit 1
fi


maxRatio=$6

if [ "$maxRatio" = "" ] ; then
  echo "Specify max. ratio of # words in docs to # of words in queries (6th arg)"
  exit 1
fi


inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"

echo "=========================================================================="
echo "Data directory:          $inputDataDir"
echo "Forward index directory: $indexDir"
echo "=========================================================================="

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[2]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

partPref=$inputDataDir/$part

scripts/data/run_export_bitext.sh -fwd_index_dir $indexDir \
                                  -index_field $field \
                                  -query_field $query_field \
                                  -output_dir "$outDir" \
                                  -q "$partPref/$queryFileName" \
                                  -qrel_file "$partPref/$QREL_FILE" \
                                  -max_doc_query_qty_ratio "$maxRatio"

