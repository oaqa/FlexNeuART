#!/bin/bash
BIN_GALAGO_SCRIPT="$1"
if [ "$BIN_GALAGO_SCRIPT" = "" ] ; then
  echo "Specify the full path to the galago execution script (1st arg)"
  exit 1
fi
if [ ! -f "$BIN_GALAGO_SCRIPT" ] ; then
  echo "Not a file: $BIN_GALAGO_SCRIPT (1st arg)"
  exit 1
fi
IN_DIR="$2"
if [ "$IN_DIR" = "" ] ; then
  echo "Specify input dir (2d arg)"
  exit 1
fi
if [ ! -d "$IN_DIR" ] ; then
  echo "Input directory: $IN_DIR doesn't exist"
  exit 1
fi

OUT_DIR="$3"
if [ "$OUT_DIR" = "" ] ; then
  echo "Specify output dir (3d arg)"
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
 "stemmedPostings" : true,
 "server" : true,
 "stemmer": ["krovetz"],
 "port" : 8080,
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

bash -c "$BIN_GALAGO_SCRIPT build $conf"
stat=$?
rm -f $conf
if [ "$stat" != "0" ] ; then
  echo "Indexing failed!!!"
  exit 1
fi

