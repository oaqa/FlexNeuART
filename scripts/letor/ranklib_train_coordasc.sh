#!/bin/bash
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

norm_type="$4"
norm=""
if [ "$norm_type" != "" ] ; then
  norm="-norm $norm_type"
fi
echo "Train file: '$train_file' Model file: '$model_file' Normalization parameter: $norm"

java -jar lib/RankLib.jar -train  "${train_file}" -ranker 4  -metric2t P@1 -gmax 1 -save "${model_file}" $norm -r $rand_rest
