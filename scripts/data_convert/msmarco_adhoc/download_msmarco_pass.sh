#!/bin/bash -e
# Downloading and re-packing the MSMARCO passage collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source scripts/common_proc.sh

cd "$dstDir"

fn=top1000.dev
wget https://msmarco.blob.core.windows.net/msmarcoranking/$fn.tar.gz
# The downloaded file contains documents too, but we only need queries.
tar zxvf $fn.tar.gz
execAndCheck "cat $fn|cut -f 1,3|sort -u > queries.devTop1000.tsv"
rm $fn.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar zxvf collection.tar.gz
gzip collection.tar

wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar zxvf queries.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
