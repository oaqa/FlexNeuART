#!/bin/bash
SUB_DIR=input/squad.orig
for f in $(ls $SUB_DIR) ;  do
  scripts/data/run_squad_reader.sh "$SUB_DIR/$f" $1
  if [ "$?" != "0" ] ; then
    echo "Failed to print $f" 1>&2 
    exit 1
  fi
done
