#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow"
  exit 1
fi

out_file=$2
if [ "$out_file" = "" ] ; then
  echo "Specify the output file (2d arg)"
  exit 1
fi


INPUT=""
for part in dev1 dev2 test train tran ; do
  INPUT="$INPUT -input_file output/$collect/$part/SolrAnswerFile.txt"
done

scripts/data/run_convert_to_trectext.sh $INPUT -output_file "$out_file"
