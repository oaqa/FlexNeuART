#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "FWD_INDEX_BACKEND_TYPES"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"

inputDataSubDir="$INPUT_DATA_SUBDIR"
indexSubDir="$FWD_INDEX_SUBDIR"

clean="0"
boolOpts=("h"     "help"    "print help"
          "clean" "clean"   "remove all previous indices")

paramOpts=("input_subdir" "inputDataSubDir" "input data sub-directory (default $inputDataSubDir)"
           "index_subdir" "indexSubDir"      "index subdirectory (default $indexSubDir)"
)

FIELD_LIST_DEF="e.g., \"text:parsedBOW text_unlemm:parsedText text_raw:textRaw dense_embed:binary\""

# This seems to be the only (though hacky way to pass space separted quoted arguments)
parseArguments "$1" "$2" "$3" $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}

usageMain="<collection> <fwd index backend: $FWD_INDEX_BACKEND_TYPES> <field list def: $FIELD_LIST_DEF>"

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

fwdIndexBackendType=${posArgs[1]}
if [ "$fwdIndexBackendType" = "" ] ; then
  genUsage "$usageMain" "Specify forward index backend type (2d arg), $FWD_INDEX_BACKEND_TYPES"
  exit 1
fi

fieldListDef="${posArgs[2]}"
if [ "$fieldListDef" = "" ] ; then
  genUsage "$usageMain" "Specify a *QUOTED* space-separated list of field index definitions (3d arg), $FIELD_LIST_DEF"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "ANSWER_FILE_BIN"

inputDataDir="$COLLECT_ROOT/$collect/$inputDataSubDir"
indexDir="$COLLECT_ROOT/$collect/$indexSubDir"

echo "=========================================================================="
echo "Input data directory:      $inputDataDir"
echo "Forward index directory:   $indexDir"
echo "Clean old index?:          $clean"
if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  if [ "$clean" = "1" ] ; then
    echo "Removing previously created index (if exists)"
    rm -rf "$indexDir"/*
  fi
fi

echo "Field list definition:     $fieldListDef"
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

for fieldDef in $fieldListDef ; do
  fieldDefSplit=(`echo $fieldDef|sed 's/:/ /'`)
  field=${fieldDefSplit[0]}
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

  target/appassembler/bin/BuildFwdIndexApp  \
    -fwd_index_backend_type $fwdIndexBackendType \
    -fwd_index_field_type $fwdIndexFieldType \
    -input_data_dir "$inputDataDir"  \
    -index_dir "$indexDir" \
    -data_sub_dirs "$dirList" \
    -data_file "$dataFileNameCurr" \
    -field_name "$field"
done

