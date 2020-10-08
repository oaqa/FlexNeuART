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

luceneIndexDir="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR/"

if [ -d "$luceneIndexDir" ] ; then
  echo "=========================================================================="
  echo "Lucene index directory:   $luceneIndexDir"

  for f in "$luceneIndexDir"/* ; do
    echo "Catting $f to /dev/null"
    time cat "$f" > /dev/null
  done
  
fi

fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"

echo "=========================================================================="
echo "Forward index directory:   $fwdIndexDir"
if [ ! -d "$fwdIndexDir" ] ; then
  echo "Directory does not exit: $fwdIndexDir"
  exit 1
fi

for f in "$fwdIndexDir"/* ; do
  echo "Catting $f to /dev/null"
  if [ -f "$f" ] ; then
    time cat "$f" > /dev/null
  fi
  if [ -d "$f" ] ; then
    for f1 in "$f"/* ; do
      echo "Catting $f1 to /dev/null"
      time cat "$f1" > /dev/null
    done
  fi
done

