#!/bin/bash -e

dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

cd "$dstDir"

fileName="psgs_w100.tsv.gz"
fileNameSuccess=${fileName}.SUCCESS

if [ -f "$fileNameSuccess" ] ; then
  echo "Already downloaded!"
  exit 0
fi

wget "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/$fileName"
touch "$fileNameSuccess"
