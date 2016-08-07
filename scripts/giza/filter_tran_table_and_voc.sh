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

FILTER_DIR="$4"
if [ "$FILTER_DIR" = "" ] ; then
  echo "Specify directory with a 'filtering' forward index, e.g., memfwdindex/compr as the 4th argument"
  exit 1
fi
if [ ! -d "$FILTER_DIR" ] ; then
  echo "$FILTER_DIR is not a directory!"
  exit 1
fi

MIN_PROB="$5"
if [ "$MIN_PROB" = "" ] ; then
  echo "Specify the minimum translation probability as the 5th argument"
  exit 1
fi

MAX_WORD_QTY="$6"
if [ "$MAX_WORD_QTY" = "" ] ; then
  echo "Specify the maximum number of (most frequent) words to use as the 6th argument"
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

function get_file_name {
  base=$1
  cnt=0
  if [ -f "$base.bz2" ] ; then
    if [ "$cnt" = "0" ] ; then
      echo "$base.bz2" 
    fi
    cnt=$(($cnt+1))
  fi
  if [ -f "$base.gz" ] ; then
    if [ "$cnt" = "0" ] ; then
      echo "$base.gz" 
    fi
    cnt=$(($cnt+1))
  fi
  if [ -f "$base" ] ; then
    if [ "$cnt" = "0" ] ; then
      echo "$base" 
    fi
    cnt=$(($cnt+1))
  fi
  if [ "$cnt" = "0" ] ; then
    echo "Not file $base, $base.gz, $base.bz2 found!"
    exit 1
  fi
  if [ "$cnt" -gt 1 ] ; then
    echo "Several versions of the $base file found (e.g., compressed and uncompressed)!"
    exit 1
  fi
}

function do_filter() {
  echo "Filtering vocabularies : '$dirSrc' -> '$dirDst'"
  SOURCE_VCB_FILE=`get_file_name "$dirSrc/source.vcb"`
  echo "Source vocabulary file: $SOURCE_VCB_FILE"
  scripts/giza/run_filter_voc.sh -i "$SOURCE_VCB_FILE"  -o  "$dirDst/source.vcb"  -memindex "$FILTER_DIR/$FIELD" -max_word_qty $MAX_WORD_QTY
  check "filter_voc source"

  TARGET_VCB_FILE=`get_file_name "$dirSrc/target.vcb"`
  echo "Target vocabulary file: $TARGET_VCB_FILE"
  scripts/giza/run_filter_voc.sh -i "$TARGET_VCB_FILE" -o  "$dirDst/target.vcb"  -memindex "$FILTER_DIR/$FIELD" -max_word_qty $MAX_WORD_QTY
  check "filter_voc target"

  echo "Filtering translation tables : '$dirSrc' -> '$dirDst'"

  echo "Translation table file: $TRAN_TABLE_FILE"
  scripts/giza/run_filter_tran_table.sh -o  "$dirDst/output.t1.${ITER_QTY}" -giza_root_dir "$dirSrc" -giza_iter_qty $ITER_QTY -min_prob "$MIN_PROB" -memindex "$FILTER_DIR/$FIELD"  -max_word_qty "$MAX_WORD_QTY"
  check "filter_tran_table"
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


