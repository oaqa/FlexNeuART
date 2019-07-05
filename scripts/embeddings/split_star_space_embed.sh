#!/bin/bash -e
in=$1
[ ! -z "$in" ] || { echo "Specify the input file (1st arg)" ; exit 1 ; }
out_prefix=$2
[ ! -z "$out_prefix" ] || { echo "Specify the output prefix (2d arg)" ; exit 1 ; }
grep --text -v '__label__' $in > ${out_prefix}.query
grep --text '__label__' $in | sed 's/__label__//' > ${out_prefix}.answer
