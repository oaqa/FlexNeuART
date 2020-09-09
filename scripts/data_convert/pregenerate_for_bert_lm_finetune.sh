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

boolOpts=(\
"h" "help" "print help"
)

seed=0
epochQty=1

paramOpts=(\
"epoch_qty" "epochQty" "# of epochs (default $epochQty)"
"seed" "seed" "seed (default $seed)"
)

parseArguments $@

usageMain="<collection> <part to be used, e.g., $BITEXT_SUBDIR>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit
fi

part=${posArgs[1]}
if [ "$part" = "" ] ; then
  echo "$usageMain" "Specify a part to be used (2d arg), e.g., $BITEXT_SUBDIR"
  exit 1
fi

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
outLMDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$LM_FINETUNE_SUBDIR"

echo "=========================================================================="
echo "Data directory:            $inputDataDir"
echo "# of epochs:               $epochQty"
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
  # The choice of the model type only affects tokenization, 
  # (given that the case is fixed) which is the same for 
  # BERT large and small
  scripts/data_convert/pregenerate_training_data.py \
    --seed $seed \
    --epochs_to_generate $epochQty \
    --bert_model bert-base-uncased \
    --train_corpus "$setFile" \
    --output_dir "$outDir"

  bzip2 $setFile
  setDirId=$(($setDirId+1))
done
