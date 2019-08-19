#!/bin/bash
source scripts/config.sh
source scripts/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "GIZA_SUBDIR"

echo "========================================================================================================================="
echo " This script assumes that each sub-directory of the directory 'tran' has two versions, e.g., 'text' and 'text.orig'"
echo " The 'text.orig' keeps data to be filtered. The filtering result is stored in the 'text' directory."
echo "========================================================================================================================="

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

field="$2"
if [ "$field" = "" ] ; then
  echo "Specify a field name: text (2d arg)"
  exit 1
fi

filterDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR"

minProb="$3"
if [ "$minProb" = "" ] ; then
  echo "Specify the minimum translation probability (3d arg)"
  exit 1
fi

maxWordQty="$4"
if [ "$maxWordQty" = "" ] ; then
  echo "Specify the maximum number of (most frequent) words to use (4th arg)"
  exit 1
fi

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
  scripts/giza/run_filter_voc.sh -i "$SOURCE_VCB_FILE"  -o  "$dirDst/source.vcb"  -flt_fwd_index_header "$filterDir/$field" -max_word_qty $maxWordQty
  check "filter_voc source"

  TARGET_VCB_FILE=`get_file_name "$dirSrc/target.vcb"`
  echo "Target vocabulary file: $TARGET_VCB_FILE"
  scripts/giza/run_filter_voc.sh -i "$TARGET_VCB_FILE" -o  "$dirDst/target.vcb"  -flt_fwd_index_header "$filterDir/$field" -max_word_qty $maxWordQty
  check "filter_voc target"

  echo "Filtering translation tables : '$dirSrc' -> '$dirDst'"

  echo "Translation table file: $TRAN_TABLE_FILE"
  scripts/giza/run_filter_tran_table.sh -o  "$dirDst/output.t1.${GIZA_ITER_QTY}" -giza_root_dir "$dirSrc" -giza_iter_qty $GIZA_ITER_QTY -min_prob "$minProb" -flt_fwd_index_header "$filterDir/$field"  -max_word_qty "$maxWordQty"
  check "filter_tran_table"
}

dirSrc="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR/$field.orig"
dirDst="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$GIZA_SUBDIR/$field"

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


