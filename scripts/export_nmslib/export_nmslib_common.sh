#!/bin/bash -e

source ./config.sh
source ./common_proc.sh


checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "MODEL1_SUBDIR"


fwdIndexDir="$FWD_INDEX_SUBDIR"
model1SubDir="$DERIVED_DATA_SUBDIR/$MODEL1_SUBDIR"
embedSubDir="$DERIVED_DATA_SUBDIR/$EMBED_SUBDIR"


params=""


boolOpts=("h" "help" "print help")


paramOpts=(
  "fwd_index_dir"   "fwdIndexDir"   "index subdirectory (default $fwdIndexDir)"
  "model1_dir"      "model1SubDir"  "a root dir for Model 1 translation fiels (e.g., GIZA output)"
  "embed_dir"       "embedSubDir"   "a root dir for embeddings"
  "query_file_pref" "queryFilePref" "query file prefix relative to input data directory (without a dot): \
If specified, we generate queries rather than documents."
  "model_file"      "modelFile"     "Linear-model file used to compute a fusion score (we don't need it for queries)."
)

parseArguments $@

usageMain="<$SAMPLE_COLLECT_ARG> <extractor JSON> <output file: relative to $DERIVED_DATA_SUBDIR>"


if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

collectDir="${COLLECT_ROOT}/$collect"
params="-collect_dir $collectDir $params"

extrJson=${posArgs[1]}
if [ "$extrJson" = "" ] ; then
  genUsage "$usageMain" "extractor JSON (2d arg)"
  exit 1
fi
params="-extr_json $extrJson $params"

outRelPath=${posArgs[2]}
if [ "$outRelPath" = "" ] ; then
  genUsage "$usageMain" "output file relative path (3d arg)"
  exit 1
fi
outFile="$collectDir/$DERIVED_DATA_SUBDIR/$outRelPath"
params="$params -out_file $outFile "

if [ "$model1SubDir" != "" ] ; then
  params="$params -model1_dir $model1SubDir"
fi

if [ "$embedSubDir" != "" ] ; then
  params="$params -embed_dir $embedSubDir"
fi

if [ "$fwdIndexDir" != "" ] ; then
  params="$params -fwd_index_dir $fwdIndexDir"
fi

if [ "$modelFile" != "" ] ; then
  params="$params -model_file $modelFile "
fi

echo "=========================================================================="
echo "Collection directory:      $collectDir                                    "
echo "Extractor JSON:            $extrJson                                      "
echo "Output file:               $outFile                                       "
echo "Forward index directory:   $fwdIndexDir                                   "
echo "Model 1 directory:         $model1SubDir                                  "
echo "Embedding directory:       $embedSubDir                                   "
echo "Model file:                $modelFile                                     "
echo "=========================================================================="

