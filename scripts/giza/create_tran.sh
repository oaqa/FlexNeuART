#!/bin/bash

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

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

FIELD=$5

if [ "$FIELD" = "" ] ; then
  echo "Specify a FIELD: e.g., text, text_unlemm, bigram, srl, srl_lab, dep, wnss (5th arg)"
  exit 1
fi

GIZA_DIR="$6"

if [ "$GIZA_DIR" = "" ] ; then
 echo "Specify GIZA/MGIZA dir (6th arg)"
 exit 1
fi

if [ ! -d "$GIZA_DIR" ] ; then
  "$GIZA_DIR isn't a directory"
  exit 1
fi

VOC_ONLY="$7"

if [ "$VOC_ONLY" = "" ] ; then
  echo "Specify VOC_ONLY (1 or 0) (7th arg)"
  exit 1
fi

stepQ=""

if [ "$VOC_ONLY" != "1" ]  ; then
  stepQ="$8"
  if [ "$stepQ" = "" ] ; then
    echo "Specify the number of iterations (8th arg)"
    exit 1
  fi
fi
 

export source_dir="$PIPELINE_OUT_PREFIX/$SUBSET/$PART"
export target_dir="$TARGET_DIR_PREFIX/$SUBSET/$PART"

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



cat ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} > "$full_target_dir/source"
check "cat ${source_dir}/answer_${FIELD}   ${source_dir}/question_${FIELD} > $full_target_dir/source"
cat ${source_dir}/question_${FIELD} ${source_dir}/answer_${FIELD}   > "$full_target_dir/target"
check "cat ${source_dir}/question_${FIELD} ${source_dir}/answer_${FIELD}   > $full_target_dir/target"

if [ "$VOC_ONLY" = "1" ] ; then
  scripts/giza/create_only_voc.sh "$GIZA_DIR" $dir "$full_target_dir/source" "$full_target_dir/target" $stepQ 
  check "create_only_voc.sh"
else
  scripts/giza/run_giza.sh "$GIZA_DIR" $dir "$full_target_dir/source" "$full_target_dir/target" $stepQ 
  check "run_giza.sh"
  rm `ls $dir/*|grep -v output.t1.$stepQ|grep -v source.vcb|grep -v target.vcb|grep -v output.gizacfg|grep -v output.perp|grep -v output.Decoder.config`
fi

rm -f $target_dir/source $target_dir/target
