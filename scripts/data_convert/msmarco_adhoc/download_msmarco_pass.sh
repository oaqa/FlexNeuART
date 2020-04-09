#!/bin/bash -e
# Downloading and re-packing the MSMARCO passage collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source scripts/common_proc.sh

cd "$dstDir"

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar zxvf queries.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar zxvf collection.tar.gz
gzip collection.tar
