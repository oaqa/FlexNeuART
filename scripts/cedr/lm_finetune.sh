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

startSetId="0"
paramOpts=(\
"grad_accum_steps" "gradAccumSteps" "# of gradient accumulation steps"
"epoch_qty" "epochQty" "# of epochs"
"batch_size" "batchSize" "batch size"
"device_name" "deviceName" "cuda device"
"grad_checkpoint_param" "gradCheckpointParam" "gradient checkpointing param"
"start_set_id" "startSetId" "start set id (use to skip initial sets)"
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

if [ "$deviceName" = "" ] ; then
  deviceName="cuda:0"
fi

  # Without fp16 the batch size needs to be 32
  #batchSize=64
  #fp16flags=" --fp16 "
if [ "$batchSize" = "" ] ; then
  batchSize=32
fi

if [ "$epochQty" = "" ] ; then
  epochQty="1"
fi

if [ "$bertLarge" = "1" ] ; then
  bertModel="bert-large-uncased"

  if [ "$gradAccumSteps" = "" ] ; then
    gradAccumSteps=1
  fi
  # BERT large is too large for Titan 1080ti
  if [ "$gradCheckpointParam" = "" ] ; then
    gradCheckpointParam="2"
  fi
else
  bertModel="bert-base-uncased"

  if [ "$gradAccumSteps" = "" ] ; then
    gradAccumSteps=1
  fi
  if [ "$gradCheckpointParam" = "" ] ; then
    gradCheckpointParam="0"
  fi
fi

outLMDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$LM_FINETUNE_SUBDIR"

modelDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR/$bertModelTopSubDir/$bertModel"

echo "=========================================================================="
echo "Output directory:          $outLMDir"
echo "Batch size:                $batchSize"
echo "CUDA device:               $deviceName"
echo "# of grad. accum. steps:   $gradAccumSteps"
echo "Grad. checkpoint param:    $gradCheckpointParam"
echo "# of epochs:               $epochQty"
echo "=========================================================================="

if [ ! -d "$modelDir" ] ; then
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

  if [ "$setId" -gt "0" ] ; then
    prevSetId=$(($setId-1))
    weightParam=" --initial_bert_weights  $modelDir/set$prevSetId/pytorch_model.bin"
  fi

  if [ "$setId" -ge "$startSetId" ] ; then

    # Note the lowercaseness
    scripts/cedr/finetune_on_pregenerated.py \
        --do_lower_case \
        --device_name $deviceName \
        --grad_checkpoint_param $gradCheckpointParam \
        --gradient_accumulation_steps $gradAccumSteps \
        --pregenerated_data "$lmDataDir" \
        --epochs $epochQty \
        $fp16flags \
        --train_batch_size $batchSize \
        --bert_model "$bertModel" \
        $weightParam \
        --output_dir "$outDir"
  else
    echo "Skip set; $setId (start set ID $startSetId)"
  fi

  setId=$(($setId+1))
done
