#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow"
  exit 1
fi

INPUT=""
for part in dev1 dev2 test train tran ; do
  INPUT="$INPUT -input_file output/$collect/$part/SolrAnswerFile.txt"
done

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

OUTPUT_DIR="output_trectext/$collect/"
if [ ! -d $OUTPUT_DIR ] ; then
  mkdir -p "$OUTPUT_DIR"
  check "mkdir -p $OUTPUT_DIR"
fi

scripts/data/run_convert_to_trectext.sh $INPUT -output_file "$OUTPUT_DIR/answers_split" -batch_qty 10000
