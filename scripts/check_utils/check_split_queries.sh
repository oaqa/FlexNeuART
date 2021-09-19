#!/bin/bash -e
. ./common_proc.sh
. ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

boolOpts=(\
"h" "help" "print help"
)

seed=0

paramOpts=()

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <input part, e.g., $BITEXT_SUBDIR> <comma-separated partition names>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

inputPart=${posArgs[1]}
if [ "$inputPart" = "" ] ; then
  genUsage "$usageMain" "Specify input part (2d arg)"
  exit 1
fi

partNames=${posArgs[2]}
if [ "$partNames" = "" ] ; then
  genUsage "$usageMain" "Specify partition names (3rd arg)"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

./check_utils/check_split_queries.py \
  --src_dir "$inputDataDir/$inputPart" \
  --dst_dir "$inputDataDir" \
  --partitions_names "$partNames"
