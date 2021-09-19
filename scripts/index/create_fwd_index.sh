#!/bin/bash -e
source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"

inputDataSubDir="$INPUT_DATA_SUBDIR"
indexSubDir="$FWD_INDEX_SUBDIR"

fwdIndexTypeArg=""
fwdIndexStoreTypeArg=""
expectedDocQtyArg=""

clean="0"
boolOpts=("h"     "help"    "print help"
          "clean" "clean"   "remove the previous index")

paramOpts=("input_subdir"           "inputDataSubDir"     "input data sub-directory (default $inputDataSubDir)"
           "index_subdir"           "indexSubDir"         "index subdirectory (default $indexSubDir)"
           "fwd_index_type"         "fwdIndexType"        "forward index type: dataDict, offsetDict"
           "fwd_index_store_type"   "fwdIndexStoreType"   "a forward backend storage type: lucene, mapdb"
           "expect_doc_qty"         "expectedDocQty"      "expected # of documents in the index"
)

FIELD_DEF="examples: text:parsedBOW, text_unlemm:parsedText, text_raw:textRaw, dense_embed:binary"

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <field definition: $FIELD_DEF>"

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

fieldDef="${posArgs[1]}"
if [ "$fieldDef" = "" ] ; then
  genUsage "$usageMain" "Specify a field definition (2d arg), $FIELD_DEF"
  exit 1
fi


if [ "$fwdIndexType" != "" ] ; then
  fwdIndexTypeArg=" -fwd_index_type $fwdIndexType "
fi
if [ "$fwdIndexStoreType" != "" ] ; then
  fwdIndexStoreTypeArg=" -fwd_index_store_type $fwdIndexStoreType "
fi
if [ "$expectedDocQty" != "" ] ; then
  expectedDocQtyArg=" -expect_doc_qty $expectedDocQty "
fi


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "ANSWER_FILE_BIN"

inputDataDir="$COLLECT_ROOT/$collect/$inputDataSubDir"
indexDir="$COLLECT_ROOT/$collect/$indexSubDir"

echo "=========================================================================="
echo "Input data directory:      $inputDataDir"
echo "Forward index directory:   $indexDir"
echo "Remove old index?:         $clean"
echo "Field definition:          $fieldDef"
echo "Index type arguments:      $fwdIndexTypeArg $fwdIndexStoreTypeArg $expectedDocQtyArg"
echo "=========================================================================="
retVal=""
getIndexQueryDataDirs "$inputDataDir"
dirList=${retVal[0]}
dataFileNameJSONL=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileNameJSONL" = "" ] ; then
  echo "Cannot find the JSONL data file"
  exit 1
fi

fieldDefSplit=(`echo $fieldDef|sed 's/:/ /'`)
field=${fieldDefSplit[0]}

if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  if [ "$clean" = "1" ] ; then
    echo "Removing previously created index for the field ${field} (if exists)"
    rm -rf "$indexDir"/"${field}"
    rm -rf "$indexDir"/"${field}."*
  fi
fi

fwdIndexFieldType=${fieldDefSplit[1]}
if [ "$fwdIndexFieldType" = "raw" ] ; then
  # For compatibility with old scripts
  fwdIndexFieldType="textRaw"
  echo "WARNING: The raw text type name is changed. Use the name 'textRaw' to avoid this warning."
fi
if [ "$field" = "" -o "$fwdIndexFieldType" = "" ] ; then
  echo "Invalid field definition $fieldDef (should be two colon-separated values, e.g, text:parsedBOW)"
  exit 1
fi

# This APP can require a lot of memory
NO_MAX=0
setJavaMem 1 8 $NO_MAX

if [ "$fwdIndexFieldType" = "binary" ] ; then
  dataFileNameCurr="$ANSWER_FILE_BIN"
else
  dataFileNameCurr="$dataFileNameJSONL"
fi

BuildFwdIndexApp  \
  $fwdIndexTypeArg \
  $fwdIndexStoreTypeArg \
  $expectedDocQtyArg \
  -fwd_index_field_type $fwdIndexFieldType \
  -input_data_dir "$inputDataDir"  \
  -index_dir "$indexDir" \
  -data_sub_dirs "$dirList" \
  -data_file "$dataFileNameCurr" \
  -field_name "$field"

