#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow"
  exit 1
fi

IN_FILE="output_trectext/$collect/answers.txt"
OUT_DIR="galago_index/$collect"

if [ ! -d "$OUT_DIR" ] ; then
  echo "Directory (or a link to a directory) $OUT_DIR doesn't exist"
  exit 1
fi

if [ ! -f "$IN_FILE" ] ; then
  echo "Input file: $IN_FILE doesn't exist"
  exit 1
fi

echo "=========================================================================="
echo "Input file:       $IN_FILE"
echo "Output directory: $OUT_DIR"
echo "Removing previous index (if exists)"
rm -f "$OUT_DIR"/*
echo "=========================================================================="

scripts/index/run_galago.sh build --indexPath=$OUT_DIR --inputPath+$IN_FILE --nonStemmedPostings=true --stemmedPostings=false --mode=threaded
if [ "$?" != "0" ] ; then
  echo "FAILURE!!!"
  exit 1
fi

