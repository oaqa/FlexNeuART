#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "DEFAULT_TRAIN_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "QUESTION_FILE_JSONL"

numCoresCPU=`getNumCpuCores`
check "getting the number of CPU cores, do you have /proc/cpu/info?"
threadQty=$numCoresCPU

boolOpts=("h" "help" "print help")

candProvOpts=""
outputFile=""

paramOpts=(
"thread_qty"             "threadQty"           "# of threads"
"out_file"               "outputFile"          "An optional output file: by default we will override existing QRELs"
"cand_prov"              "candProv"            "Candidate record provider type"
"cand_prov_uri"          "providerURI"         "Provider URI: an index location, a query server address, etc"
"cand_prov_add_conf"     "candProvAddConf"     "JSON with additional candidate provider parameters"
)

usageMain="<$SAMPLE_COLLECT_ARG> <part type> <field name> <top-K candidates to use>"

parseArguments $@

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

trainPart=${posArgs[1]}
if [ "$trainPart" = "" ] ; then
  genUsage "$usageMain" "Specify training part, e.g., $DEFAULT_TRAIN_SUBDIR (2d arg)"
  exit 1
fi

fieldName=${posArgs[2]}
if [ "$fieldName" = "" ] ; then
  genUsage "$usageMain" "Specify the field name (2d arg)"
  exit 1
fi

candQty=${posArgs[3]}
if [ "$candQty" = "" ] ; then
  echo "Specify a number of candidate entries, e.g., 1000  (4th arg)"
  exit 1
fi

if [ "$providerURI" = "" ] ; then
  providerURI="$LUCENE_INDEX_SUBDIR/"
fi

collectDir="$COLLECT_ROOT/$collect"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR/$trainPart"
fwdIndexDir="$FWD_INDEX_SUBDIR/"

candProvOpts=" -u $providerURI "

if [ "$candProvAddConf" != "" ] ; then
  candProvOpts="-cand_prov_add_conf $candProvAddConf $candProvOpts "
fi

if [ "$candProv" != "" ] ; then
  candProvOpts="-cand_prov $candProv $candProvOpts "
fi

if [ "$outputFile" = "" ] ; then
  outputFile="$inputDataDir/$QREL_FILE"
fi

queryFileName="$inputDataDir/$QUESTION_FILE_JSONL"

echo "=========================================================================="
echo "Collection directory:      $collectDir"
echo "Data directory:            $inputDataDir"
echo "Output file:               $outputFile"
echo "Candidate provider options:$candProvOpts"
echo "Field name:                $fieldName"
echo "Forward index directory:   $fwdIndexDir"
echo "Query file name:           $queryFileName"
echo "# of threads:              $threadQty"
echo "=========================================================================="


target/appassembler/bin/AnswerBasedQRELGenerator \
    -collect_dir $collectDir \
    $candProvOpts \
    -cand_qty $candQty \
    -thread_qty $threadQty \
    -fwd_index_dir $fwdIndexDir \
    -out_file "$outputFile"  \
    -q  "$queryDir/$queryFileName"

