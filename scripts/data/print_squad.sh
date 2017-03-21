#!/bin/bash
for f in `ls input/squad/|grep -v titles|grep -v interm|grep -v ner` ;  do
  scripts/data/run_squad_reader.sh input/squad/$f $1
done
