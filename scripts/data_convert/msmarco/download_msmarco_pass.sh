#!/bin/bash -e
# Downloading and re-packing the MSMARCO passage collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source ./common_proc.sh

cd "$dstDir"

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar zxvf collectionandqueries.tar.gz
gzip collection.tsv

for year in 2019 2020 ; do
  wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test${year}-queries.tsv.gz
  gunzip msmarco-test${year}-queries.tsv.gz
done

wget https://trec.nist.gov/data/deep/2019qrels-pass.txt
wget https://trec.nist.gov/data/deep/2020qrels-pass.txt

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
tar zxvf queries.tar.gz

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv

