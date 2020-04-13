#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

boolOpts=(\
"h" "help" "print help"
)

paramOpts=(\
"part1_qty" "part1qty" "1st part # of entries" \
"part1_fract" "part1fract" "1st part fraction # of entries" \
)

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <input subdir> <1st output subdir> <2d output subdir>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

inputSubdir=${posArgs[1]}
if [ "$inputSubdir" = "" ] ; then
  genUsage "$usageMain" "Specify input subdir (2d arg)"
  exit 1
fi

outSubdir1=${posArgs[2]}
if [ "$outSubdir1" = "" ] ; then
  genUsage "$usageMain" "Specify 1st output subdir (3d arg)"
  exit 1
fi

outSubdir2=${posArgs[3]}
if [ "$outSubdir1" = "" ] ; then
  genUsage "$usageMain" "Specify 1st output subdir (4th arg)"
  exit 1
fi

if [ "$part1qty" != "" ] ; then
  qtyParam="--part1_qty $part1qty"
elif [ "$part1fract" != "" ] ; then
  qtyParam="--part1_fract $part1fract"
else
  echo "Specify one of the following: -part1_qty or -part1_fract!"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

scripts/data_convert/split_queries.py \
--data_dir "$inputDataDir" \
--input_subdir $inputSubdir --out_subdir1 $outSubdir1 --out_subdir2 $outSubdir2 \
$qtyParam
