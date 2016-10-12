#!/bin/bash
test_file="$1"
if [ "$test_file" = "" ] ; then
  echo "Specify test feature-file (1st arg)"
  exit 1
fi
if [ ! -f "$test_file" ] ; then
  echo "Not a file: '$test_file'" 
  exit 1
fi

model_file="$2"
if [ "$model_file" = "" ] ; then
  echo "Specify model file (2d arg)"
  exit 1
fi
if [ ! -f "$model_file" ] ; then
  echo "Not a file: '$model_file'" 
  exit 1
fi
norm_type="$3"
norm=""
if [ "$norm_type" != "" ] ; then
  norm="-norm $norm_type"
fi
echo "Test file: '$test_file' Model file: '$model_file' Normalization parameter: $norm"

java -jar lib/RankLib.jar -test "$test_file"   -metric2t P@1 -gmax 1 -load "$model_file"  $norm
