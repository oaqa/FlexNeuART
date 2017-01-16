#!/bin/bash
. scripts/common.sh

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
    if [ "$d" = "wiki" ] ; then
      cd $cdir
      check "cd $cdir"
      # In this case, we simply create a link, because for the wiki part there are no annotated questions
      ln -s SolrAnswerFile.txt SolrQuestionAnswerFile.txt
      check "ln -s SolrAnswerFile.txt SolrQuestionAnswerFile.txt"
      cd -
      check "cd -"
    else
      cat $cdir/SolrQuestionFile.txt $IN_DIR/$d/SolrAnswerFile.txt > $cdir/SolrQuestionAnswerFile.txt 
      check "$cdir/SolrQuestionFile.txt $IN_DIR/$d/SolrAnswerFile.txt > $cdir/SolrQuestionAnswerFile.txt "
    fi
  done
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

