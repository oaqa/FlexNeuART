#!/bin/bash -e
# Based on https://github.com/facebookresearch/DPR/blob/master/data/download_data.py
datasetName=$1;

if [ "$datasetName" = "" ] ; then
  echo "Specify datasetName as a first argument: nq, trivia, squad (1st arg)";
  exit 1;
fi

dstDir=$2
[ "$dstDir" != "" ] || { echo "Specify destination directory (2d arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (2d arg)" ; exit 1 ; }

cd "$dstDir"

case $datasetName in

   nq)
     train="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz";
     dev="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz";
     test="";
     ;;

   trivia)
     train="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz";
     dev="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz";
     test="";
     ;;

   squad)
     train="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz";
     dev="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz";
     test="";
     ;;

   *)
     echo "Incorrect dataset name: $datasetName Available options: [nq, trivia, squad]"; exit 1;
     ;;
esac

wget $train -O $datasetName\_train.json.gz
wget $dev -O $datasetName\_dev.json.gz
