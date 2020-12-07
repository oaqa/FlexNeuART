#!/bin/bash -e

dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

cd "$dstDir"

wget "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
