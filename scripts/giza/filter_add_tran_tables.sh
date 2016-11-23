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

FIELD="text"
ITER_QTY="5"
MIN_PROB="2.5e-3"

COL=$1

if [ "$COL" = "" ] ; then
  print "Specify the collection (e.g., compr, stackoverflow) as the 1st arg"
  exit 1
fi

for d in  tran/$COL/tran* 
do 
  if [ -d $d ] ; then
    echo "Processing folder $d with minimum translation probability: $MIN_PROB"
    scripts/giza/simple_filter_tran_table.sh "$d" $FIELD $ITER_QTY $MIN_PROB
    check "scripts/giza/simple_filter_tran_table.sh "$d" $FIELD $ITER_QTY $MIN_PROB"
  fi
done
