#!/bin/bash
IN_DIR="$1"
if [ "$IN_DIR" = "" ] ; then
  echo "Specify input dir (1st arg)"
  exit 1
fi
if [ ! -d "$IN_DIR" ] ; then
  echo "Input directory: $IN_DIR doesn't exist"
  exit 1
fi

OUT_DIR="$2"
if [ "$OUT_DIR" = "" ] ; then
  echo "Specify output dir (2d arg)"
  exit 1
fi

if [ ! -d "$OUT_DIR" ] ; then
  echo "Directory (or a link to a directory) $OUT_DIR doesn't exist"
  exit 1
fi

conf=`mktemp`
cat > $conf <<EOF
{
 "inputPath": "$IN_DIR",
 "indexPath" : "$OUT_DIR",
 "nonStemmedPostings" : false,
 "server" : true,
 "stemmer": "krovetz",
 "port" : 8080,
 "stemmedPostings" : true,
 "mode" : "local" 
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

scripts/index/galago build $conf
stat=$?
rm -f $conf
if [ "$stat" != "0" ] ; then
  echo "Indexing failed!!!"
  exit 1
fi

