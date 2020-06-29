#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SUBDIR"
checkVarNonEmpty "LM_FINETUNE_SET_PREF"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

#--bert_large

# A convenience wrapper for the corresponding Python script,
# which "knows" locations of collection sub-directories.

boolOpts=(\
"h" "help" "print help"
"bert_large" "bertLarge" "use LARGE BERT model"
)

parseArguments $@

usageMain="<collection> <BERT model sub-directory>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit
fi

bertModelTopSubDir=${posArgs[1]}
if [ "$bertModelTopSubDir" = "" ] ; then
  echo "$usageMain" "Specify BERT model sub-directory root (2d arg)"
  exit 1
fi

if [ "$bertLarge" = "1" ] ; then
  initModel="bert-large-uncased"
  # Without fp16 the batch size needs to be 32
  #batchSize=64
  #fp16flags=" --fp16 "
  batchSize=16
else
  initModel="bert-base-uncased"
  # Without fp16 the batch size needs to be 32
  #batchSize=64
  #fp16flags=" --fp16 "
  batchSize=32
fi

outLMDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$LM_FINETUNE_SUBDIR"

modelDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$bertModelTopSubDir/$initModel"

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
