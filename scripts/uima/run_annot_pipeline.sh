#!/bin/bash

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, ComprMinusManner, stackoverflow"
  exit 1
fi

if [ "$collect" = "manner" ] ; then
  for d in Dev1 Dev2 Train Test ; do 
    echo $d 
    scripts/uima/cpe.sh src/main/resources/descriptors/collection_processing_engines/cpeAnnotManner${d}.xml 2>&1|tee log_$d
    if [ "$?" != "0" ] ; then
      echo "FAILURE!!!"
      exit 1
    fi
  done
elif [ "$collect" = "compr" ] ; then
  for d in Dev1 Dev2 Train1 Train2 Tran Test ; do 
    echo $d 
    scripts/uima/cpe.sh src/main/resources/descriptors/collection_processing_engines/cpeAnnotCompr${d}.xml 2>&1|tee log_$d
    if [ "$?" != "0" ] ; then
      echo "FAILURE!!!"
      exit 1
    fi
  done
elif [ "$collect" = "stackoverflow" ] ; then
  for d in Dev1 Dev2 Train Tran Test ; do 
    echo $d 
    scripts/uima/cpe.sh src/main/resources/descriptors/collection_processing_engines/cpeAnnotStackOverflow${d}.xml 2>&1|tee log_$d
    if [ "$?" != "0" ] ; then
      echo "FAILURE!!!"
      exit 1
    fi
  done
elif [ "$collect" = "ComprMinusManner" ] ; then
  scripts/uima/cpe.sh src/main/resources/descriptors/collection_processing_engines/cpeAnnotComprMinusMannerTran.xml 2>&1|tee log_ComprMinusManner
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

