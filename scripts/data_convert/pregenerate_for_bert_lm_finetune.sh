#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SUBDIR"
checkVarNonEmpty "BITEXT_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SET_PREF"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

# A convenient wrapper for the corresponding Python script

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

part=$2
if [ "$part" = "" ] ; then
  echo "Specify a part to be used (1st arg), e.g., $BITEXT_SUBDIR"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
outLMDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$LM_FINETUNE_SUBDIR"

echo "=========================================================================="
echo "Data directory:            $inputDataDir"
echo "Output directory:          $outLMDir"
echo "=========================================================================="

if [ -d "$outLMDir" ] ; then
  echo "Removing previously created data (if exists)"
  rm -rf "$outLMDir"/*
else
  mkdir -p "$outLMDir"
fi
retVal=""

getIndexQueryDataInfo "$inputDataDir"
dataFileName=${retVal[1]}
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

inputFile="$inputDataDir/$part/$dataFileName"

if [ ! -f "$inputFile" ] ; then
  echo "Input file: $inputFile does not exist, perhaps, you specified the wrong collection part"
  exit 1
fi

scripts/data_convert/convert_to_bert_lm_finetune_data.py \
  --input "$inputFile"  \
  --lower_case  \
  --output_pref "$outLMDir/$LM_FINETUNE_SET_PREF"

setDirId=0
for setFile in "$outLMDir/$LM_FINETUNE_SET_PREF"* ; do

  outDir="$outLMDir/${LM_FINETUNE_SUBDIR}_pregen$setDirId"

  scripts/pytorch-transformers/pregenerate_training_data.py \
    --epochs_to_generate 1 --bert_model bert-base-uncased \
    --train_corpus "$setFile" \
    --output_dir "$outDir"

  bzip2 $setFile
  setDirId=$(($setDirId+1))
done
