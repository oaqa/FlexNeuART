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
saveEpochSnapshots=0
saveEpochSnapshotsArg=""

boolOpts=("h" "help" "print help"
          "save_epoch_snapshots" "saveEpochSnapshots" "save snapshot after each epoch")

seed=0
epochQty=1
batchesPerEpoch=0
masterPort=10001
deviceQty=1
batchSyncQty=4
deviceName="cuda:0"
addExperSubdir=""
jsonConf=""
initModelWeights=""
initModel=""
bertLarge="0"
vocabFile=""
optim="adamw"
momentum="0.9"
validCheckPoints=""
validRunDir=""
maxQueryVal=""

paramOpts=("seed"          "seed"             "seed (default $seed)"
      "optim"              "optim"            "optimizer (default $optim)"
      "momentum"           "momentum"         "SGD momentum (default $momentum)"
      "epoch_qty"          "epochQty"         "# of epochs (default $epochQty)"
      "batches_per_train_epoch"  "batchesPerEpoch"  "# of batches per train epoch (default $batchesPerEpoch)"
      "max_query_val"      "maxQueryVal"      "max # of val queries"
      "valid_checkpoints"  "validCheckPoints" "validation checkpoints (in # of batches)"
      "valid_run_dir"      "validRunDir"      "directory to store full predictions on validation set"
      "master_port"        "masterPort"       "master port for multi-GPU train (default $masterPort)"
      "device_name"        "deviceName"       "device name for single-gpu train (default $deviceName)"
      "device_qty"         "deviceQty"        "# of device (default $deviceQty)"
      "batch_sync_qty"     "batchSyncQty"     "# of batches before model sync"
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
  exit 1
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
  echo "Cleaning up the output directory: $outModelDir"
  # A safer way to clean is to make sure that:
  # 1. the variable is not empty and it is a directory
  # 2. go there, then delete.
  # 3. otherwise we risk doing rm -rf /  *
  checkVarNonEmpty "outModelDir"
  # -d will fail for empty name as well
  if [ ! -d "$outModelDir" ] ; then
    echo "$Not a directory: $outModelDir"
    exit 1
  else
    pushd "$outModelDir"
    # In the worst case we will delete files in the current directory only
    rm -rf ./*
    popd
  fi
fi

collectDir="$COLLECT_ROOT/$collect"

jsonConfArg=""

if [ "$saveEpochSnapshots" = "1" ] ; then
  saveEpochSnapshotsArg=" --save_epoch_snapshots "
fi

validCheckPointsArg=""
if [ "$validCheckPoints" != "" ] ; then
  validCheckPointsArg=" --valid_checkpoints $validCheckPoints "
fi

validRunDirArg=""
if [ "$validRunDir" != "" ] ; then
  validRunDirArg=" --valid_run_dir $outModelDir/$validRunDir "
fi

maxQueryValArg=""
if [ "$maxQueryVal" != "" ] ; then
  maxQueryValArg=" --max_query_val $maxQueryVal "
fi

echo "=========================================================================="
echo "Training data directory:                        $trainDir"
echo "Output model directory:                         $outModelDir"
echo "# of epochs:                                    $epochQty"
echo "Save snapshots arg:                             $saveEpochSnapshotsArg"
echo "BERT large?:                                    $bertLarge"
echo "seed:                                           $seed"
echo "device #:                                       $deviceQty"
echo "# of batches before model sync:                 $batchSyncQty"
echo "optimizer:                                      $optim"
echo "validation checkpoints arg:                     $validCheckPointsArg"
echo "validation run dir  arg:                        $validRunDirArg"
echo "batches per train epoch:                        $batchesPerEpoch"
echo "max # of valid. queries arg:                    $maxQueryValArg"

if [ "$deviceQty" = "1" ] ; then
  echo "device name:                                    $deviceName"
else
  echo "master port:                                    $masterPort"
fi

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

python -u scripts/cedr/train.py \
  $initModelArgs \
  $jsonConfArg \
  $bertLargeArg \
  $vocabFileArg \
  $validCheckPointsArg \
  $validRunDirArg \
  $maxQueryValArg \
  --optim $optim \
  --momentum $momentum \
  --seed $seed \
  --device_name $deviceName \
  --device_qty $deviceQty \
  --batch_sync_qty $batchSyncQty \
  --epoch_qty $epochQty \
  $saveEpochSnapshotsArg \
  --batches_per_train_epoch $batchesPerEpoch \
  --master_port $masterPort \
  --datafiles "$trainDir/data_query.tsv"  \
              "$trainDir/data_docs.tsv" \
   --train_pairs "$trainDir/train_pairs.tsv" \
  --valid_run "$trainDir/test_run.txt" \
  --qrels "$trainDir/qrels.txt" \
  --model_out_dir "$outModelDir" \
2>&1|tee "$outModelDir/train.log"

