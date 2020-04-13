#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"

if [ ! -d "$outDir" ] ; then
  mkdir -p "$outDir"
fi

field=$2

if [ "$field" = "" ] ; then
  echo "Specify a document/index field: e.g., text_unlemm (2d arg)"
  exit 1
fi

query_field=$3

if [ "$query_field" = "" ] ; then
  echo "Specify a query field, e.g., text (3d arg)"
  exit 1
fi

maxRatio=$4

if [ "$maxRatio" = "" ] ; then
  echo "Specify max. ratio of # words in docs to # of words in queries (4th arg)"
  exit 1
fi


echo "=========================================================================="
echo "Data directory:          $inputDataDir"
echo "Forward index directory: $indexDir"
echo "Embedding directory:     $embedDir"
echo "Max ratio:               $maxRatio"
echo "=========================================================================="

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

partPref=$inputDataDir/$BITEXT_SUBDIR

target/appassembler/bin/CreateBitextFromQRELs -fwd_index_dir $indexDir \
                                  -embed_dir $embedDir \
                                  -index_field $field \
                                  -query_field $query_field \
                                  -output_dir "$outDir" \
                                  -q "$partPref/$queryFileName" \
                                  -qrel_file "$partPref/$QREL_FILE" \
                                  -max_doc_query_qty_ratio "$maxRatio"

