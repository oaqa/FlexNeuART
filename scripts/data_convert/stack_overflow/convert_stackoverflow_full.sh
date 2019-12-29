#!/bin/bash
. scripts/common_proc.sh

input="$1"

if [ "$input" = "" ] ; then
  echo "Specify input file (1st arg)"
  exit 1
fi

interm="$2"

if [ "$interm" = "" ] ; then
  echo "Specify intermediate file prefix (2d arg)"
  exit 1
fi

output="$3"

if [ "$output" = "" ] ; then
  echo "Specify output file (3d arg)"
  exit 1
fi


scripts/data/run_convert_stackoverflow_step1.sh -input "$input" -output "${interm}1" 
check "scripts/data/run_convert_stackoverflow_step1.sh -input \"$input\" -output \"${interm}1\" "

sort -k 1,1 -n -s "${interm}1" > "${interm}2"
check "sort -k 1,1 -n -s \"$input\" > \"${interm}2\""

scripts/data/run_convert_stackoverflow_step2.sh -input "${interm}2" -output "$output" -exclude_code
check "scripts/data/run_convert_stackoverflow_step2.sh -input \"${interm}2\" -output \"$output\" -exclude_code"
