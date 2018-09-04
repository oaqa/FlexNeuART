#!/bin/bash -e
in=$1
out_prefix=$2
grep -v '__label__' $in > ${out_prefix}.query
grep '__label__' $in | sed 's/__label__//' > ${out_prefix}.answer
