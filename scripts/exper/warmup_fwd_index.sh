#!/bin/bash -e
# Just a silly script to warm up forward index,
# b/c otherwise it sometimes takes too long to
# warm up through the normal course of re-ranking
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"

indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"

echo "=========================================================================="
echo "Forward index directory:   $indexDir"
if [ ! -d "$indexDir" ] ; then
  echo "Directory does not exit: $indexDir"
  exit 1
fi

for f in "$indexDir"/* ; do
  echo "Catting $f to /dev/null"
  time cat "$f" > /dev/null
done

