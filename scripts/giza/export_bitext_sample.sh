#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "BITEXT_TRAIN_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$EMBED_SUBDIR/"

if [ ! -d "$outDir" ] ; then
  mkdir "$outDir"
fi

field=$2

if [ "$field" = "" ] ; then
  echo "Specify a document/index field: e.g., text (2d arg)"
  exit 1
fi

query_field=$3

if [ "$query_field" = "" ] ; then
  echo "Specify a query field, e.g., text (3d arg)"
  exit 1
fi

sampleQty=$4

if [ "$sampleQty" = "" ] ; then
  echo "Specify # of samples (4th arg)"
  exit 1
fi

embedFile=$5

if [ "$embedFile" = "" ] ; then
  echo "Specify embeddings file relative to the embedding dir (5th arg)"
  exit 1
fi

echo "=========================================================================="
echo "Data directory:          $inputDataDir"
echo "Forward index directory: $indexDir"
echo "Embedding directory:     $embedDir"
echo "=========================================================================="

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[2]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

partPref=$inputDataDir/$BITEXT_TRAIN_SUBDIR

scripts/data/run_export_bitext.sh -fwd_index_dir "$indexDir" \
                                  -embed_dir "$embedDir" \
                                  -embed_file "$embedDir/$embedFile" \
                                  -index_field $field \
                                  -query_field $query_field \
                                  -output_dir "$outDir" \
                                  -q "$partPref/$queryFileName" \
                                  -qrel_file "$partPref/$QREL_FILE" \
                                  -sample_qty "$sampleQty"

