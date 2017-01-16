#!/bin/bash

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg) squad"
  exit 1
fi

IN_DIR="output/$collect/"

if [ ! -d "$IN_DIR" ] ; then
  echo "The directory '$IN_DIR' doesn't exist!"
  exit 1
fi

echo "=========================================================================="
echo " Working directory: $IN_DIR"
echo "=========================================================================="

if [ "$collect" = "squad" ] ; then
  for d in train dev1 dev2 test tran wiki  ; do
    cdir=$IN_DIR/$d
    echo "Merging in $cdir"
    cat $cdir/SolrQuestionFile.txt $IN_DIR/$d/SolrAnswerFile.txt > $cdir/SolrQuestionAnswerFile.txt 
    if [ "$?" != "0" ] ; then
     echo "FAILURE!!!"
     exit 1
    fi
  done
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

