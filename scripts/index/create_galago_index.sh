#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow"
  exit 1
fi

IN_DIR="output_trectext/$collect/"
OUT_DIR="galago_index/$collect"

if [ ! -d "$OUT_DIR" ] ; then
  echo "Directory (or a link to a directory) $OUT_DIR doesn't exist"
  exit 1
fi

if [ ! -d "$IN_DIR" ] ; then
  echo "Input directory: $IN_DIR doesn't exist"
  exit 1
fi

conf=`mktemp`
cat > $conf <<EOF
{
 "inputPath": "$IN_DIR",
 "indexPath" : "$OUT_DIR",
 "nonStemmedPostings" : true,
 "stemmedPostings" : false,
 "mode" : "threaded" 
}
EOF

echo "=========================================================================="
echo "Input directory:       $IN_DIR"
echo "Output directory: $OUT_DIR"
echo "Removing previous index (if exists)"
rm -rf "$OUT_DIR"/*
echo "Config file:"
cat $conf
echo "=========================================================================="

../galago-3.10-bin/bin/galago build $conf
stat=$?
rm -f $conf
if [ "$stat" != "0" ] ; then
  echo "Indexing failed!!!"
  exit 1
fi

