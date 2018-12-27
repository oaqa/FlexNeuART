#!/bin/bash
. scripts/common.sh

PIPELINE_OUT_PREFIX="$1"

if [ "$PIPELINE_OUT_PREFIX" = "" ] ; then
 echo "Specify the pipeline output top-level directory (1st arg)"
 exit 1
fi

if [ ! -d "$PIPELINE_OUT_PREFIX" ] ; then
  "$PIPELINE_OUT_PREFIX isn't a directory"
  exit 1
fi

TARGET_DIR_PREFIX="$2"

if [ "$TARGET_DIR_PREFIX" = "" ] ; then
 echo "Specify the top-level directory for translation files (2d arg)"
 exit 1
fi

if [ ! -d "$TARGET_DIR_PREFIX" ] ; then
  "$TARGET_DIR_PREFIX isn't a directory"
  exit 1
fi

SUBSET=$3

if [ "$SUBSET" = "" ] ; then
  echo "Specify a SUBSET: e.g., compr, stackoverflow, ComprMinusManner (3d arg)"
  exit 1
fi

PART=$4

if [ "$PART" = "" ] ; then
  echo "Specify a PART: e.g., train, dev1, dev2, test, tran (4th arg)"
  exit 1
fi

SYMMETRIZE=$5

if [ "$SYMMETRIZE" != "0" -a "$SYMMETRIZE" != "1" ] ; then
  echo "Specify a symmetrization flag : 0 or 1 (5th arg)"
  exit 1
fi

TARGET_SUBDIR=$6

if [ "$TARGET_SUBDIR" = "" ] ; then
  echo "Specify a sub-directory in the target directory: e.g., full, part2, part4, part8, ... (6th arg)"
  exit 1
fi

FIELD=$7

if [ "$FIELD" = "" ] ; then
  echo "Specify a FIELD: e.g., text, text_unlemm, bigram, srl, srl_lab, dep, wnss (7th arg)"
  exit 1
fi

GIZA_DIR="$8"

if [ "$GIZA_DIR" = "" ] ; then
 echo "Specify GIZA/MGIZA dir (8th arg)"
 exit 1
fi

if [ ! -d "$GIZA_DIR" ] ; then
  "$GIZA_DIR isn't a directory"
  exit 1
fi

VOC_ONLY="$9"

if [ "$VOC_ONLY" = "" ] ; then
  echo "Specify VOC_ONLY (1 or 0) (9th arg)"
  exit 1
fi

stepQ=""

if [ "$VOC_ONLY" != "1" ]  ; then
  stepQ="${10}"
  if [ "$stepQ" = "" ] ; then
    echo "Specify the number of iterations (9th arg)"
    exit 1
  fi
fi

echo "SUBSET: $SUBSET PART: SYMMETRIZE: $SYMMETRIZE $PART TARGET_DIR: $TARGET_SUBDIR FIELD: $FIELD VOC_ONLY=$VOC_ONLY # of iterations: $stepQ"
 

export source_dir="$PIPELINE_OUT_PREFIX/$SUBSET/$PART"
export target_dir="$TARGET_DIR_PREFIX/$SUBSET/$TARGET_SUBDIR"

echo "Source dir prefix=$source_dir" 
echo "Target dir prefix=$target_dir" 

if [ ! -d "$target_dir" ] ; then
  echo "Creating: $dir"
  mkdir -p $target_dir 
  check "mkdir -p $target_dir"
fi


export dir="$target_dir/${FIELD}.orig"
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

# Note that answers are the source corpus and questions are the target one!
# This is also confirmed by the paper:
# Finding Similar Questions in Large Question and Answer Archives
# Jiwoon Jeon, W. Bruce Croft and Joon Ho Lee 
# However, a better performance is achieved with a symmetrized approach.

cd $target_dir
check "cd $target_dir"
full_target_dir="$PWD"
cd -

echo "Full target dir: $full_target_dir"


# Non-symmetric version commented out
#cat ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} > "$full_target_dir/source"
#check "cat ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} > $full_target_dir/source"
#cat ${source_dir}/question_${FIELD} ${source_dir}/answer_${FIELD}   > "$full_target_dir/target"
#check "cat ${source_dir}/question_${FIELD} ${source_dir}/answer_${FIELD}   > $full_target_dir/target"

# Symmetric version commented out
#cat ${source_dir}/question_${FIELD}    > "$full_target_dir/target"
#check "cat ${source_dir}/question_${FIELD}    > $full_target_dir/target"
#cat ${source_dir}/answer_${FIELD}  > "$full_target_dir/source"
#check "cat ${source_dir}/answer_${FIELD}  > $full_target_dir/source"

# A unified version that also filters out sentences where the difference in the number of words is too large 
MAX_FERTILITY=9
scripts/giza/filter_long.py ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} $MAX_FERTILITY "$full_target_dir/source" "$full_target_dir/target" $SYMMETRIZE
check "scripts/giza/filter_long.py ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} $MAX_FERTILITY $full_target_dir/source $full_target_dir/target $SYMMETRIZE"

if [ "$VOC_ONLY" = "1" ] ; then
  scripts/giza/create_only_voc.sh "$GIZA_DIR" $dir "$full_target_dir/source" "$full_target_dir/target" $stepQ 
  check "create_only_voc.sh"
else
  scripts/giza/run_mgiza.sh "$GIZA_DIR" $dir "$full_target_dir/source" "$full_target_dir/target" $stepQ 
  check "run_giza.sh"
  rm `ls $dir/*|grep -v output.t1.$stepQ|grep -v source.vcb|grep -v target.vcb|grep -v output.gizacfg|grep -v output.perp|grep -v output.Decoder.config`
fi

rm -f $target_dir/source $target_dir/target
