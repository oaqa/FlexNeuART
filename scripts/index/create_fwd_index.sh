#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi

fieldListDef=$2
if [ "$fieldListDef" = "" ] ; then
  echo "Specify a *QUOTED* space-separated list of field index definitions (2d arg), e.g., 'text:parsedBOW text_unlemm:parsedText text_raw:raw'"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
indexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"

echo "=========================================================================="
echo "Data directory:            $inputDataDir"
echo "Forward index directory:   $indexDir"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previously created index (if exists)"
  rm -rf "$indexDir"/*
fi

echo "Field list definition:     $fieldListDef"
echo "=========================================================================="
retVal=""
getIndexQueryDataInfo "$inputDataDir"
dirList=${retVal[0]}
dataFileName=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

for fieldDef in $fieldListDef ; do
  fieldDefSplit=(`echo $fieldDef|sed 's/:/ /'`)
  field=${fieldDefSplit[0]}
  fwdIndexType=${fieldDefSplit[1]}
  if [ "$field" = "" -o "$fwdIndexType" = "" ] ; then
    echo "Invalid field definition $fieldDef (should be two colon-separated values, e.g, text:parsedBOW)"
    exit 1
  fi
  scripts/index/run_buildfwd_index.sh \
    -fwd_index_type $fwdIndexType \
    -input_data_dir "$inputDataDir"  \
    -index_dir "$indexDir" \
    -data_sub_dirs "$dirList" \
    -data_file "$dataFileName" \
    -field_name "$field"
done

