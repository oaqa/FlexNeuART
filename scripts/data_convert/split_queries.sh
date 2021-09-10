#!/bin/bash -e
source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

boolOpts=(\
"h" "help" "print help"
)

seed=0

paramOpts=(\
"seed"          "seed"        "random seed, default ($seed)"
)

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <input part, e.g., $BITEXT_SUBDIR> \
<comma-separated partition names> <comma-separated partition sizes: empty means all remaining>"

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

partSizes=${posArgs[3]}
if [ "$partSizes" = "" ] ; then
  genUsage "$usageMain" "Specify partition size (4th arg)"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

./data_convert/split_queries.py \
  --seed $seed \
  --src_dir "$inputDataDir/$inputPart" \
  --dst_dir "$inputDataDir" \
  --partitions_names "$partNames" \
  --partitions_sizes "$partSizes"
