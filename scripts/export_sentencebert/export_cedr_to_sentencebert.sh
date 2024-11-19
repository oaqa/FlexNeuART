#!/bin/bash -e

source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"

parseArguments $@

usageMain="<collection> <train data subdir (relative to derived data)> <output data subdir (relative to derived data)>"

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}

if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

derivedDataDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR"

trainSubDir=${posArgs[1]}

if [ "$trainSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify cedr training data subdir relative to $derivedDataDir (2nd arg)"
  exit 1
fi

outSubDir=${posArgs[2]}

if [ "$outSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify sentence bert training data subdir relative to $derivedDataDir (3rd arg)"
  exit 1
fi

echo "=========================================================================="

set -o pipefail
trainDir="$derivedDataDir/$trainSubDir"
sbertDir="$derivedDataDir/$outSubDir"

python -u ./export_sentencebert/export_cedr_to_sentencebert.py --datafiles "$trainDir/data_query.tsv"  \
              "$trainDir/data_docs.tsv" \
   --train_pairs "$trainDir/train_pairs.tsv" \
   --valid_run "$trainDir/test_run.txt" \
   --qrels "$trainDir/qrels.txt" \
   --output_dir_name $sbertDir 
