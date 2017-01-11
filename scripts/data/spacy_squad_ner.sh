#!/bin/bash
. scripts/common.sh

DIR=$1

if [ "$DIR" = "" ] ; then
  echo "Specify the input directory (1st arg)"
  exit 1
fi

if [ ! -d "$DIR" ] ; then
  echo "Not a directory (1st arg)"
  exit 1
fi

THREAD_QTY=$2

if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify the number of threads (2d arg)"
  exit 1
fi

for f in $DIR/*-interm*.gz ; do
  tf=`echo $f|sed s/interm/spacy_ner/`
  echo "$f -> $tf"
  scripts/data/run_spacy_ner.py $f $tf $THREAD_QTY
  check "scripts/data/run_spacy_ner.py $f $tf $THREAD_QTY"
done

