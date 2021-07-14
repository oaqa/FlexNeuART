#!/bin/bash -e
# A script to derive Model 1 translation probabilities using MGIZA.

. scripts/common_proc.sh
. scripts/config.sh


checkVarNonEmpty "SAMPLE_COLLECT_ARG"

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "MODEL1_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

sampleProb=1

boolOpts=(\
"h" "help" "print help"
)

paramOpts=(
  "bitext_subdir" "bitextSubDir"  "bitext sub-dir, if not specified we use $BITEXT_SUBDIR"
  "model1_subdir" "model1SubDir"  "GIZA sub-dir to store translation table, if not specified we use $MODEL1_SUBDIR"
  "sample_prob"   "sampleProb"    "sample probability (default $sampleProb)"
)

usageMain="<collection> <field>"

parseArguments $@

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

if [ "$model1SubDir" = "" ] ; then
  model1SubDir=$MODEL1_SUBDIR
fi

if [ "$bitextSubDir" = "" ] ; then
  bitextSubDir=$BITEXT_SUBDIR
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit
fi

field=${posArgs[1]}
if [ "$field" = "" ] ; then
  genUsage "$usageMain" "Specify a field: e.g., text (2d arg)"
  exit 1
fi

mgizaDir="$PWD/mgiza"

if [ ! -d "$mgizaDir" ] ; then
  "$mgizaDir isn't a directory"
  exit 1
fi

VOC_ONLY="0"
SYMMETRIZE=1 # Nearly always works better

export source_dir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$bitextSubDir"
export target_dir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$model1SubDir"

echo "=========================================================================="
echo "GIZA (output) sub-directory:  $model1SubDir"
echo "Bitext sub-directory:         $bitextSubDir"
echo "Source dir prefix:            $source_dir"
echo "Target dir prefix:            $target_dir"
echo "Sample probability:           $sampleProb"
echo "=========================================================================="

if [ ! -d "$source_dir" ] ; then
  echo "Directory does not exist: $source_dir!"
  exit 1
fi

if [ ! -d "$target_dir" ] ; then
  echo "Creating: $dir"
  mkdir -p "$target_dir"
fi

export dir="$target_dir/${field}.orig"
echo "Dir=$dir" 
if [ ! -d "$dir" ] ; then
  echo "Creating: $dir"
  mkdir $dir 
  check "mkdir $dir "
else
  echo "Cleaning up: '$dir'"
  rm -f $dir/*
  check "rm -f $dir/*"
fi

cd $target_dir
check "cd $target_dir"
full_target_dir="$PWD"
cd -

set pipefile

echo "Full target dir: $full_target_dir"

#
# Note that answers are the source corpus and questions are the target one:
# after all, IBM Model 1 needs conditional probabilities p(q|a)
# and not the other way around.
#
# This is also confirmed by the paper:
# Finding Similar Questions in Large Question and Answer Archives
# Jiwoon Jeon, W. Bruce Croft and Joon Ho Lee 
# However, a better performance is achieved with a symmetrized approach,
# where for each pair (q, a) we also add a flipped pair (a, q)
#

# 1. Filtering out sentences where the difference in the number of words is too large 
# 2. Applying symmetrization if requested
MAX_FERTILITY=9
scripts/giza/sample_and_filter_long_bitext.py "${source_dir}/answer_${field}"   \
                            "${source_dir}/question_${field}" \
                            "$MAX_FERTILITY" \
                            "$full_target_dir/source" "$full_target_dir/target" \
                            "$SYMMETRIZE" \
                            "$sampleProb"

scripts/giza/run_mgiza.sh "$mgizaDir" "$dir" \
                          "$full_target_dir/source" "$full_target_dir/target" \
                          "$GIZA_ITER_QTY"

pushd "$full_target_dir/${field}.orig"
rm `ls *|grep -v output.t1.${GIZA_ITER_QTY}|grep -v source.vcb|grep -v target.vcb|grep -v output.gizacfg|grep -v output.perp|grep -v output.Decoder.config`
popd

rm -f "$target_dir/source" "$target_dir/target"
