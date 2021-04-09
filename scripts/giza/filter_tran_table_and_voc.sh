#!/bin/bash
source scripts/config.sh
source scripts/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "GIZA_ITER_QTY"
checkVarNonEmpty "GIZA_SUBDIR"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

boolOpts=(\
"h" "help" "print help"
)

paramOpts=(\
   "giza_subdir" "gizaSubDir" "GIZA sub-dir to store translation table, if not specified we use $GIZA_SUBDIR"
)

parseArguments $@

usageMain="<collection> <field> <min. probability> <max freq. word qty>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

if [ "$gizaSubDir" = "" ] ; then
  gizaSubDir=$GIZA_SUBDIR
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

filterDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR"

minProb="${posArgs[2]}"
if [ "$minProb" = "" ] ; then
  genUsage "$usageMain" "Specify the minimum translation probability (3d arg)"
  exit 1
fi

maxWordQty="${posArgs[3]}"
if [ "$maxWordQty" = "" ] ; then
  genUsage "$usageMain" "Specify the maximum number of (most frequent) words to use (4th arg)"
  exit 1
fi

dirSrc="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$gizaSubDir/$field.orig"
dirDst="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$gizaSubDir/$field"

echo "========================================================================================================================="
echo " This script uses (but doesn't modify) the data created by scripts/giza/create_tran.sh which is placed in the directory:"
echo "$dirSrc"
echo "The filtered output is stored in the following directory:"
echo "$dirDst"
echo "========================================================================================================================="

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
  target/appassembler/bin/FilterVocabulary -i "$SOURCE_VCB_FILE"  -o  "$dirDst/source.vcb"  -flt_fwd_index_header "$filterDir/$field" -max_word_qty $maxWordQty
  check "filter_voc source"

  TARGET_VCB_FILE=`get_file_name "$dirSrc/target.vcb"`
  echo "Target vocabulary file: $TARGET_VCB_FILE"
  target/appassembler/bin/FilterVocabulary -i "$TARGET_VCB_FILE" -o  "$dirDst/target.vcb"  -flt_fwd_index_header "$filterDir/$field" -max_word_qty $maxWordQty
  check "filter_voc target"

  echo "Filtering translation tables : '$dirSrc' -> '$dirDst'"

  echo "Translation table file: $TRAN_TABLE_FILE"
  target/appassembler/bin/FilterTranTable -o  "$dirDst/output.t1.${GIZA_ITER_QTY}" -giza_root_dir "$dirSrc" -giza_iter_qty $GIZA_ITER_QTY -min_prob "$minProb" -flt_fwd_index_header "$filterDir/$field"  -max_word_qty "$maxWordQty"
  check "filter_tran_table"
}

dirSrc="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$gizaSubDir/$field.orig"
dirDst="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$gizaSubDir/$field"

if [ ! -d "$dirSrc" ] ; then
  echo "Error, '$dirSrc' doesn't exist"
  exit 1
fi

if [ ! -d "$dirDst" ] ; then
  mkdir "$dirDst"
  check "mkdir \"$dirDst\""
fi

do_filter 
check "filter tran table"


