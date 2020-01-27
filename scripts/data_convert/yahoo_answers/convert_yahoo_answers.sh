#!/bin/bash -e
# The main script to convert document collection in the Yahoo Answers format,
# which as previously split by the script:
# scripts/data_convert/yahoo_answers/split_yahoo_answers_input.sh
#
# This script makes the following assumption:
#
# Data files end with <underscore> <part name> .gz  e.g., comprehensive-Oct2007_dev1.gz
#
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "INPUT_RAW_SUBDIR"

partList=$2
if [ "$partList" = "" ] ; then
  echo "Specify the list of comma-separated parts, next arg is optional bitext part (2d arg)"
  exit 1
fi
bitextPart=$3

inputRawDir="$COLLECT_ROOT/$collect/$INPUT_RAW_SUBDIR"
inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
biTextSubir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$BITEXT_SUBDIR"

if [ ! -d "$inputRawDir" ] ; then
  echo "Directory doesn't exist: $inputRawDir"
  exit 1
fi

echo "=========================================================================="
echo "Raw data directory: $inputRawDir"
echo "Processed data directory: $inputDataDir"
if [ ! -d "$inputDataDir" ] ; then
  mkdir -p "$inputDataDir"
else
  echo "Removing previous data (if exists)"
  rm -rf "$inputDataDir"/*
fi

partList=`echo $partList|sed 's/,/ /g'`

for part in $partList ; do
  dstDir="$inputDataDir/$part"
  mkdir -p "$dstDir"
  lsCmd="ls \"$inputRawDir\" 2>/dev/null | grep _${part}.gz "
  qty=`bash -c "$lsCmd" | wc -l | awk '{print $1}'`
  if [ "$qty" != "1" ] ; then
    echo "Wrong number of files for part $part: $qty (should be exactly one)"
    exit 1
  fi
  inputFile=`bash -c "$lsCmd"`
  echo $inputFile
  biTextOpt=""
  if [ "$bitextPart" = "$part" ] ; then
    if [ -d "$biTextSubir" ] ; then
      rm -f "$biTextSubir"/*
    else
      mkdir -p "$biTextSubir"
    fi
    biTextOpt="--out_bitext_path $biTextSubir"
  fi
  outMainPath="$inputDataDir"/"$part"/
  if [ -d "$outMainPath" ] ; then
    rm -f "$outMainPath"/*
  else
    mkdir "$outMainPath"
  fi
  python -u scripts/data_convert/yahoo_answers/convert_yahoo_answers.py \
            --input "$inputRawDir"/"$inputFile" \
            --out_main_path $outMainPath \
            $biTextOpt
done


