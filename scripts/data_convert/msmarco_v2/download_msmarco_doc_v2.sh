#!/bin/bash -e
# Downloading and re-packing the MSMARCO (v2) document collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source ./common_proc.sh

cd "$dstDir"

for fn in \
          docv2_train_queries.tsv \
          docv2_train_qrels.tsv \
          docv2_dev_queries.tsv \
          docv2_dev_qrels.tsv \
          docv2_dev2_queries.tsv \
          docv2_dev2_qrels.tsv \
          msmarco_v2_doc.tar \
          ; do
  uri=https://msmarco.z22.web.core.windows.net/msmarcoranking/$fn
  echo "Downloading $uri"
  # Header suggestion is from https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#downloading-the-datasets
  wget --header "X-Ms-Version: 2019-12-12"  $uri
done

tar xvf msmarco_v2_doc.tar  

