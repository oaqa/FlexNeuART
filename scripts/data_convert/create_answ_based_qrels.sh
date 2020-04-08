#!/bin/bash
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "FWD_INDEX_SUBDIR"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

candQty=$2
if [ "$candQty" = "" ] ; then
  echo "Specify a number of candidate entries, e.g., 1000  (2d arg)"
  exit 1
fi

numCoresCPU=`getNumCpuCores`
check "getting the number of CPU cores, do you have /proc/cpu/info?"
threadQty=$numCoresCPU


inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"
fwdIndexDir="$COLLECT_ROOT/$collect/$FWD_INDEX_SUBDIR/"
luceneIndexDir="$COLLECT_ROOT/$collect/$LUCENE_INDEX_SUBDIR/"
retVal=""
getIndexQueryDataInfo "$inputDataDir"
dirList=${retVal[2]}
queryFileName=${retVal[3]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
dirList=$(echo $dirList|sed 's/,/ /g')
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

echo "=========================================================================="
echo "Data directory:            $inputDataDir"
echo "Lucene index directory:    $luceneIndexDir"
echo "Forward index directory:   $fwdIndexDir"
echo "Query file name:           $queryFileName"
echo "# of threads:              $threadQty"
echo "=========================================================================="

for subDir in $dirList ; do

  queryDir=$inputDataDir/$subDir

  [ -d "$queryDir" ] || { "Missing directory $queryDir" ; exit 1 ; }

  echo "Processing sub-dir $subDir"

  scripts/data_convert/run_answ_based_qrel_gen.sh \
      -cand_qty $candQty \
      -thread_qty $threadQty \
      -fwd_index_dir $fwdIndexDir \
      -u $luceneIndexDir \
      -out_file "$queryDir/$QREL_FILE"  \
      -q  "$queryDir/$queryFileName"

done
