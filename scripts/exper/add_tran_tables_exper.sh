#!/bin/bash

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

COL=$1

if [ "$COL" = "" ] ; then
  print "Specify the collection (e.g., compr, stackoverflow) as the 1st arg"
  exit 1
fi

FIELD="text"
DPREF="tran/$COL"
EXPER_DESC_FILE="exper.cfg"
echo "exper@bm25=text+model1=text   @   dev1" > $EXPER_DESC_FILE
for n in `ls  $DPREF/|grep tran` 
do 
  d=$DPREF/$n
  if [ -d $d ] ; then
    echo "Processing folder $d"
    if [ -h "$DPREF/$FIELD" ] ; then
      rm $DPREF/$FIELD
      check "rm $DPREF/$FIELD"
    fi
    cd $DPREF
    check "cd $DPREF"
    ln -s $n/$FIELD  
    check "ln -s $n/$FIELD"
    cd -
    check "cd -"
    scripts/exper/run_feature_experiments.sh $COL "graded_same_score" $EXPER_DESC_FILE 1 "50000,10000" 2>&1
    check "scripts/exper/run_feature_experiments.sh $COL graded_same_score $EXPER_DESC_FILE 1 50000,10000 2>&1"
  fi
done
