#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

fwdIndexType=$2
if [ "$fwdIndexType" = "" ] ; then
  echo "Specify forward index type (2d arg), e.g., mapdb, lucene, flatdata"
  exit 1
fi

fieldListDef=$3
if [ "$fieldListDef" = "" ] ; then
  echo "Specify a *QUOTED* space-separated list of field index definitions (3d arg), e.g., 'text:parsedBOW text_unlemm:parsedText text_raw:raw'"
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
  fwdIndexStoreType=${fieldDefSplit[1]}
  if [ "$field" = "" -o "$fwdIndexType" = "" ] ; then
    echo "Invalid field definition $fieldDef (should be two colon-separated values, e.g, text:parsedBOW)"
    exit 1
  fi

  # This APP can require a lot of memory
  NO_MAX=0
  setJavaMem 6 8 $NO_MAX

  target/appassembler/bin/BuildFwdIndexApp  \
    -fwd_index_type $fwdIndexType \
    -fwd_index_store_type $fwdIndexStoreType \
    -input_data_dir "$inputDataDir"  \
    -index_dir "$indexDir" \
    -data_sub_dirs "$dirList" \
    -data_file "$dataFileName" \
    -field_name "$field"
done

