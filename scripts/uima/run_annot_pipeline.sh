#!/bin/bash
. scripts/common.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, ComprMinusManner, stackoverflow"
  exit 1
fi

if [ "$collect" = "manner" ] ; then
  for d in Dev1 Dev2 Train Test ; do 
    echo $d 
    uima_desc="src/main/resources/descriptors/collection_processing_engines/cpeAnnotManner${d}.xml"
    scripts/uima/cpe.sh "$uima_desc" 2>&1|tee log_$d
    check_pipe "UIMA pipeline with CPE descriptor $uima_desc"
  done
elif [ "$collect" = "compr" ] ; then
  for d in Dev1 Dev2 Train Tran Test ; do 
    echo $d 
    uima_desc="src/main/resources/descriptors/collection_processing_engines/cpeAnnotCompr${d}.xml"
    scripts/uima/cpe.sh "$uima_desc"  2>&1|tee log_$d
    check_pipe "UIMA pipeline with CPE descriptor $uima_desc"
  done
elif [ "$collect" = "stackoverflow" ] ; then
  for d in Dev1 Dev2 Train Tran Test ; do 
    echo $d 
    uima_desc="src/main/resources/descriptors/collection_processing_engines/cpeAnnotStackOverflow${d}.xml"
    scripts/uima/cpe.sh "$uima_desc"  2>&1|tee log_$d
    check_pipe "UIMA pipeline with CPE descriptor $uima_desc"
  done
elif [ "$collect" = "squad" ] ; then
  for d in Wiki Dev1 Dev2 Train Tran Test ; do 
    echo $d 
    uima_desc="src/main/resources/descriptors/collection_processing_engines/cpeAnnotSQuAD${d}.xml"
    scripts/uima/cpe.sh "$uima_desc"  2>&1|tee log_$d
    check_pipe "UIMA pipeline with CPE descriptor $uima_desc"
  done
elif [ "$collect" = "ComprMinusManner" ] ; then
  uima_desc="src/main/resources/descriptors/collection_processing_engines/cpeAnnotComprMinusMannerTran.xml"
  scripts/uima/cpe.sh "$uima_desc"  2>&1|tee log_ComprMinusManner
  check_pipe "UIMA pipeline with CPE descriptor $uima_desc"
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

