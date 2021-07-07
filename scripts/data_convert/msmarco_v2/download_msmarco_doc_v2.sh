#!/bin/bash -e
# Downloading and re-packing the MSMARCO (v2) document collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source scripts/common_proc.sh

check_has_azcopy

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
  uri=https://msmarco.blob.core.windows.net/msmarcoranking/$fn
  echo "Downloading $uri"
  azcopy copy "$uri" .
done

