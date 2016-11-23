#!/bin/bash

MIN_PROB="2.5e-3"

COL=$1

if [ "$COL" = "" ] ; then
  print "Specify the collection (e.g., compr, stackoverflow) as the 1st arg"
  exit 1
fi

for d in  tran/$COL/tran* 
do 
  echo "Processing folder $d "
done
