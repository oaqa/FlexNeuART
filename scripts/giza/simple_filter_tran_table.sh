#!/bin/bash

echo "========================================================================================================================="
echo " This script assumes that each sub-directory of the directory 'tran' has two versions, e.g., 'text' and 'text.orig'"
echo " The 'text.orig' keeps data to be filtered. The filtering result is stored in the 'text' directory."
echo "========================================================================================================================="

TRAN_PREFIX="$1"

if [ "$TRAN_PREFIX" = "" ] ; then
  echo "Specify a translation prefix, e.g., 'tran/ComprMinusManner' as the 1st argument"
  exit 1
fi
if [ ! -d "$TRAN_PREFIX" ] ; then
  echo "$TRAN_PREFIX is not a directory!"
  exit 1
fi

FIELD="$2"
if [ "$FIELD" = "" ] ; then
  echo "Specify a field name: text, bigram, ... as the 2d argument"
  exit 1
fi

ITER_QTY="$3"
if [ "$ITER_QTY" = "" ] ; then
  echo "Specify the number of GIZA iterations, e.g., 5, as the 3d argument"
  exit 1
fi

MIN_PROB="$4"
if [ "$MIN_PROB" = "" ] ; then
  echo "Specify the minimum translation probability as the 4th argument"
  exit 1
fi

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

function do_filter() {
  echo "Copying vocabularies : '$dirSrc' -> '$dirDst'"
  SOURCE_VCB_FILE="$dirSrc/source.vcb"
  echo "Source vocabulary file: $SOURCE_VCB_FILE"
  cp "$SOURCE_VCB_FILE"  "$dirDst"
  check "cp $SOURCE_VCB_FILE  $dirDst"

  TARGET_VCB_FILE="$dirSrc/target.vcb"
  echo "Target vocabulary file: $TARGET_VCB_FILE"
  cp "$TARGET_VCB_FILE"  "$dirDst"
  check "cp $TARGET_VCB_FILE  $dirDst"

  echo "Filtering translation tables : '$dirSrc' -> '$dirDst'"

  TRAN_TABLE_SRC_FILE="$dirSrc/output.t1.${ITER_QTY}"
  TRAN_TABLE_DST_FILE="$dirDst/output.t1.${ITER_QTY}"

  awk -v min_prob="$MIN_PROB" '{if ($3>=min_prob) print $0}' $TRAN_TABLE_SRC_FILE > $TRAN_TABLE_DST_FILE
  check "awk -v min_prob=$MIN_PROB '{if ($3>=min_prob) print $0}' $TRAN_TABLE_SRC_FILE > $TRAN_TABLE_DST_FILE"
}

dirSrc=$TRAN_PREFIX/$FIELD.orig
dirDst=$TRAN_PREFIX/$FIELD

if [ ! -d "$dirSrc" ] ; then
  echo "Error, '$dirDst' doesn't exist"
  exit 1
fi

if [ ! -d "$dirDst" ] ; then
  mkdir "$dirDst"
  check "mkdir \"$dirDst\""
fi

do_filter 
check "filter tran table"


