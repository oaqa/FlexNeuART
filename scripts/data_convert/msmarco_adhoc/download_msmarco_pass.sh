#!/bin/bash -e
# Downloading and re-packing the MSMARCO passage collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source scripts/common_proc.sh

cd "$dstDir"

for year in 2019 2020 ; do
  wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test${year}-queries.tsv.gz
  gunzip msmarco-test${year}-queries.tsv.gz
done

wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar zxvf queries.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar zxvf collection.tar.gz
gzip collection.tsv
