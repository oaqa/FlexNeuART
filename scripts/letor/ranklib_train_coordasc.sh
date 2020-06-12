#!/bin/bash -e
train_file="$1"
if [ "$train_file" = "" ] ; then
  echo "Specify training feature-file (1st arg)"
  exit 1
fi
if [ ! -f "$train_file" ] ; then
  echo "Not a file: '$train_file'" 
  exit 1
fi

model_file="$2"
if [ "$model_file" = "" ] ; then
  echo "Specify output file for the model (2d arg)"
  exit 1
fi

rand_rest="$3"
if [ "$rand_rest" = "" ] ; then
  echo "Specify the number of random restarts (3d arg)"
  exit 1
fi

metric_type="$4"
if [ "$metric_type" = "" ] ; then
  echo "Specify the metric type (4th arg)"
  exit 1
fi

echo "Train file: '$train_file' Model file: '$model_file' Metric type: $metric_type"

DIM_NUM_ITERS=50 # Twice as much as the default

java -jar lib/RankLib.jar -train  "${train_file}" -ranker 4  -save "${model_file}" \
     -metric2t $metric_type  -metric2T $metric_type  \
     -r $rand_rest -i $DIM_NUM_ITERS
