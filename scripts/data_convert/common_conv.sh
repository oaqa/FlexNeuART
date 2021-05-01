#!/usr/bin/env bash
source scripts/config.sh
source scripts/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"

src=$1
if [ "$src" = "" ] ; then
  echo "Specify the source directory (1st arg)"
  exit 1
fi
collect=$2
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (2d arg)"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

echo "Source data directory: $src"
echo "Target data directory: $inputDataDir"
