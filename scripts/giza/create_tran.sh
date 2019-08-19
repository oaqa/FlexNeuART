#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

field=$2

if [ "$field" = "" ] ; then
  echo "Specify a field: e.g., text (2d arg)"
  exit 1
fi

mgizaDir="$3"

if [ "$mgizaDir" = "" ] ; then
 echo "Specify MGIZA dir (3d arg)"
 exit 1
fi

if [ ! -d "$mgizaDir" ] ; then
  "$mgizaDir isn't a directory"
  exit 1
fi

VOC_ONLY="0"
SYMMETRIZE=1 # Nearly always work better

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "GIZA_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_ITER_QTY"

export source_dir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR"
export target_dir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR"

echo "Source dir prefix=$source_dir" 
echo "Target dir prefix=$target_dir" 

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

echo "Full target dir: $full_target_dir"

# Note that answers are the source corpus and questions are the target one!
# This is also confirmed by the paper:
# Finding Similar Questions in Large Question and Answer Archives
# Jiwoon Jeon, W. Bruce Croft and Joon Ho Lee 
# However, a better performance is achieved with a symmetrized approach.

# 1. Filtering out sentences where the difference in the number of words is too large 
# 2. Applying symmetrizing if needed
MAX_FERTILITY=9
execAndCheck "scripts/giza/filter_long.py \"${source_dir}/answer_${field}\"   \"${source_dir}/question_${field}\" \"$MAX_FERTILITY\" \"$full_target_dir/source\" \"$full_target_dir/target\" \"$SYMMETRIZE\""

execAndCheck "scripts/giza/run_mgiza.sh \"$mgizaDir\" $dir \"$full_target_dir/source\" \"$full_target_dir/target\" $GIZA_ITER_QTY"

rm `ls $dir/*|grep -v output.t1.$stepQ|grep -v source.vcb|grep -v target.vcb|grep -v output.gizacfg|grep -v output.perp|grep -v output.Decoder.config`

rm -f $target_dir/source $target_dir/target
