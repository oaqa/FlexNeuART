#!/bin/bash -e
# This script creates a parallel corpus from a set of queries, documents, and QRELs.
# Because computing translation probabilities via expectation-maximization
# requires parallel chunks of text to be of a similar length, documents are
# split into chunks and the query is repeated paired with each such chunk.
# There is a parameter defining how much longer document chunks can be compared
# to queries (in terms of the relative number of words)

. ./common_proc.sh
. ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "BITEXT_SUBDIR"

bitextInSubDir=$BITEXT_SUBDIR
bitextOutSubDir=$BITEXT_SUBDIR

boolOpts=(\
"h" "help" "print help"
)

paramOpts=(\
"bitext_in_subdir" "bitextInSubDir" "bitext input sub-dir (default $bitextInSubDir)"
"bitext_out_subdir" "bitextOutSubDir" "bitext targed sub-dir (default $bitextOutSubDir)"
)

parseArguments $@

usageMain="<collection> <index field> <query field> <max query to doc word ratio>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

checkVarNonEmpty "bitextInSubDir"
checkVarNonEmpty "bitextOutSubDir"

collect=${posArgs[0]}

if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "EMBED_SUBDIR"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "QUESTION_FILE_PREFIX"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
outDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$bitextOutSubDir"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
embedDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR/"

if [ ! -d "$outDir" ] ; then
  mkdir -p "$outDir"
fi

field=${posArgs[1]}

if [ "$field" = "" ] ; then
  genUsage "$usageMain" "Specify a document/index field: e.g., text_unlemm (2d arg)"
  exit 1
fi

query_field=${posArgs[2]}

if [ "$query_field" = "" ] ; then
  genUsage "$usageMain" "Specify a query field, e.g., text (3d arg)"
  exit 1
fi

maxRatio=${posArgs[3]}

if [ "$maxRatio" = "" ] ; then
  genUsage "$usageMain" "Specify max. ratio of # words in docs to # of words in queries (4th arg)"
  exit 1
fi

inputPartDir="$inputDataDir/$bitextInSubDir"

echo "=========================================================================="
echo "Data directory:               $inputDataDir"
echo "Forward index directory:      $indexDir"
echo "Bitext input sub-directory:   $inputPartDir"
echo "Index field:                  $field"
echo "Query field:                  $query_field"
echo "Bitext output sub-directory:  $outDir"
echo "Embedding directory:          $embedDir"
echo "Max ratio:                    $maxRatio"
echo "=========================================================================="

CreateBitextFromQRELs -fwd_index_dir $indexDir \
          -embed_dir $embedDir \
          -index_field $field \
          -query_field $query_field \
          -output_dir "$outDir" \
          -query_file_pref  "$inputPartDir/${QUESTION_FILE_PREFIX}" \
          -qrel_file "$inputPartDir/$QREL_FILE" \
          -max_doc_query_qty_ratio "$maxRatio"

