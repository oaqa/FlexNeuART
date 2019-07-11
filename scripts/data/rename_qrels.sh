#!/bin/bash -e

outDir=$1

if [ "$outDir" = "" ] ; then
  echo "Specify the output directory, e.g., output/manner"
  exit 1
fi
if [ ! -d "$outDir" ] ; then
  echo "Not a directory: $outDir"
  exit 1
fi

cd "$outDir"

for subDir in "$outDir"/* ; do
  if [ -d "$subDir" ] ; then
      cd "$subDir"
      for f in `ls | fgrep qrels_ | fgrep -v 'qrels_all_graded.txt'` ; do
        echo rm $f
      done
      mv qrels_all_graded.txt qrels.txt
      cd ..
  fi
done
