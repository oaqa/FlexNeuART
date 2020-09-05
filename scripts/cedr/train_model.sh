#!/usr/bin/env bash
# This is a convenience wrapper that does a bit of extra heavy lifting for model training.
# In particular, based on the collection name and few options it
# 1. Identifies the key directories to read the training data and store the model
# 2. Copies the JSON configuration file to the model output directory, if such as JSON file
#    was provided
# 3. Saves the training log
set -eo pipefail


. scripts/common_proc.sh
. scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "IR_MODELS_SUBDIR"

sampleProb=1

boolOpts=("h" "help" "print help")

seed=0
epochQty=1
batchesPerEpoch=0
masterPort=10001
deviceQty=1
deviceName="cuda:0"
addExperSubdir=""
jsonConf=""
initModelWeights=""
initModel=""
bertLarge="0"
vocabFile=""
optim="adamw"
momentum="0.9"

paramOpts=("seed"          "seed"             "seed (default $seed)"
      "optim"              "optim"            "optimizer (default $optim)"
      "momentum"           "momentum"         "SGD momentum (default $momentum)"
      "epoch_qty"          "epochQty"         "# of epochs (default $epochQty)"
      "batches_per_train_epoch"  "batchesPerEpoch"  "# of batches per train epoch (default $batchesPerEpoch)"
      "master_port"        "masterPort"       "master port for multi-GPU train (default $masterPort)"
      "device_name"        "deviceName"       "device name for single-gpu train (default $deviceName)"
      "device_qty"         "deviceQty"        "# of device (default $deviceQty)"
      "add_exper_subdir"   "addExperSubdir"   "additional experimental sub-directory (optional)"
      "json_conf"          "jsonConf"         "collection relative JSON configuration file (optional)"
      "vocab_file"         "vocabFile"        "vocabulary file relative to derived-data directory (optional)"
      "init_model_weights" "initModelWeights" "initial model weights"
      "init_model"         "initModel"        "init model"
      "bert_large"         "bertLarge"        "specify 1 to use BERT"
)

parseArguments $@

usageMain="<collection> <train data subdir (relative to derived data)> <model type>"

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
fi

collect=${posArgs[0]}

if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

derivedDataDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR"

trainSubDir=${posArgs[1]}

if [ "$trainSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify training data subdir relative to $derivedDataDir (2d arg)"
  exit 1
fi

modelType=${posArgs[2]}
if [ "$modelType" = "" ] ; then
  genUsage "$usageMain" "Specify model type, e.g., vanilla_bert (3rd arg)"
  exit 1
fi

initModelArgs=""
if [ "$initModelWeights" != "" ] ; then
  initModelArgs=" --model $modelType --init_model_weights $initModelWeights "
elif [ "$initModel" != "" ] ; then
  initModelArgs=" --init_model $initModel "
else
  initModelArgs=" --model $modelType "
  echo "WARNING: neither -init_model_weights nor -init_model specified, training from random init!"
fi

bertLargeArg=""
bertTypeSubdir="base"
if [ "$bertLarge" = "1" ] ; then
  bertLargeArg="  --model.bert_large"
  bertTypeSubdir="large"
fi

outModelDir="$derivedDataDir/$IR_MODELS_SUBDIR/$modelType/$addExperSubdir/$bertTypeSubdir/$seed/"
trainDir="$derivedDataDir/$trainSubDir"


if [ ! -d "$outModelDir" ] ; then
  echo "Creating new output directory"
  mkdir -p "$outModelDir"
else
  echo "Cleaning up the output directory"
  rm -f "$outModelDir"/*
fi

jsonConfArg=""



echo "=========================================================================="
echo "Training data directory:                        $trainDir"
echo "Output model directory:                         $outModelDir"
echo "# of epochs:                                    $epochQty"
echo "BERT large?:                                    $bertLarge"
echo "seed:                                           $seed"
echo "device #:                                       $deviceQty"
echo "optimizer:                                      $optim"

if [ "$deviceQty" = "1" ] ; then
  echo "device name:                                    $deviceName"
else
  echo "master port:                                    $masterPort"
fi

collectDir="$COLLECT_ROOT/$collect"

if [ "$jsonConf" != "" ] ; then
  cp "$collectDir/$jsonConf" "$outModelDir"
  bn=`basename "$jsonConf"`
  jsonConfDest="$outModelDir/$bn"
  jsonConfArg=" --json_conf $jsonConfDest "
  echo "JSON config:                                    $jsonConfDest"
fi

if [ "$vocabFile" != "" ] ; then
  vocabFileFullPath="$derivedDataDir/$vocabFile"
  vocabFileArg=" --model.vocab_file $vocabFileFullPath "
  echo "Vocabulary file path:                           $vocabFileFullPath"
fi

echo "=========================================================================="

scripts/cedr/train.py \
  $initModelArgs \
  $jsonConfArg \
  $bertLargeArg \
  $vocabFileArg \
  --optim $optim \
  --momentum $momentum \
  --seed $seed \
  --device_name $deviceName \
  --device_qty $deviceQty \
  --epoch_qty $epochQty \
  --batches_per_train_epoch $batchesPerEpoch \
  --master_port $masterPort \
  --datafiles "$trainDir/data_query.tsv"  \
              "$trainDir/data_docs.tsv" \
   --train_pairs "$trainDir/train_pairs.tsv" \
  --valid_run "$trainDir/test_run.txt" \
  --qrels "$trainDir/qrels.txt" \
  --model_out_dir "$outModelDir" \
2>&1|tee "$outModelDir/train.log"

