#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SET_PREF"

# A convenient wrapper for the corresponding Python script

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g., squad"
  exit 1
fi
bertModelSubDir=$2
if [ "$bertModelSubDir" = "" ] ; then
  echo "Specify BERT model sub-directory"
  exit 1
fi

outLMDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$LM_FINETUNE_SUBDIR"

modelDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$bertModelSubDir"

echo "=========================================================================="
echo "Output directory:          $outLMDir"
echo "=========================================================================="

if [ -d "$modelDir" ] ; then
  echo "Removing previously created model (if exists)"
  rm -rf "$modelDir"
else
  mkdir -p "$modelDir"
fi
retVal=""

initModel="bert-base-uncased"
# Without fp16 the batch size needs to be 32
#batchSize=64
#fp16flags=" --fp16 "
batchSize=32
setId=0
for lmDataDir in "$outLMDir/${LM_FINETUNE_SUBDIR}_pregen"* ; do
  echo "Finetuning using data in $lmDataDir"

  # We would like to keep all the models
  outDir="$modelDir/set$setId"

  if [ ! -d "$outDir" ] ; then
    mkdir -p "$outDir"
  fi

  scripts/pytorch-transformers/finetune_on_pregenerated.py \
      --pregenerated_data "$lmDataDir" \
      --epochs 1 \
      $fp16flags \
      --train_batch_size $batchSize \
      --bert_model "$initModel" \
      --output_dir "$outDir" 

  setId=$(($setId+1))
  initModel="$outDir"
done
