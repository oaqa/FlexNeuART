#!/usr/bin/env bash
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source ./common_proc.sh

cd "$dstDir"

wget "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
tar zxvf cran.tar.gz

