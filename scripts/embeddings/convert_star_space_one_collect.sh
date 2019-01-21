#!/bin/bash -e
colName=$1
if [ "$colName" = "" ] ; then
  echo "Specify the collection name (1st arg)!"
  exit 1
fi
if [ -d WordEmbeddings/${colName}/starspace ] ; then
  rm -f WordEmbeddings/${colName}/starspace/*
else
  mkdir -p WordEmbeddings/${colName}/starspace
fi
for d in 100 200 400 ; do
  echo $d
  for f in WordEmbeddings/${colName}.src/starspace/$d/*.tsv ; do
    fb=$(basename $f .tsv)
    echo "$d $f : $fb"
    scripts/embeddings/split_star_space_embed.sh $f WordEmbeddings/${colName}/starspace/dim=${d}_$fb
  done
done
