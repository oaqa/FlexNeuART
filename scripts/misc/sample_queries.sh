#!/bin/bash -e
. ./common_proc.sh
. ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "QREL_FILE"

boolOpts=("h" "help" "print help")

paramOpts=("seed" "seed" "random seed")

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <input subdir> <output subdir> <# of queries>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

if [ "$seed" = "" ] ; then
  seed=0
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

outSubdir=${posArgs[2]}
if [ "$outSubdir" = "" ] ; then
  genUsage "$usageMain" "Specify output subdir (3d arg)"
  exit 1
fi

qty=${posArgs[3]}
if [ "qty" = "" ] ; then
  genUsage "$usageMain" "Specify # of queries (4th arg)"
  exit 1
fi

if [ "$inputSubdir" = "$outSubdir" ] ; then
  genUsage "$usageMain" "Output and input directories should be different!"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

inpDir="$inputDataDir/$inputSubdir"
outDir="$inputDataDir/$outSubdir"

echo "============================="
echo "Input directory:   $inpDir"
echo "Output directory:  $outDir"
echo "Seed:              $seed"
echo "# of queries:      $qty"
echo "============================="

./data_convert/sample_queries.py \
  --data_dir "$inputDataDir" \
  --input_subdir $inputSubdir \
  --out_subdir $outSubdir --qty $qty
