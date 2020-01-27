#!/bin/bash
# A very basic script to obtain basic collection stat.
# This would need to be turned into a fully-fledged
# stat script that would read JSON question/answer files
# and produce more fine-grained data


source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
echo "getIndexQueryDataInfo return value: " ${retVal[*]}
indexDirs=`echo ${retVal[0]}|sed 's/,/ /g'`
dataFileName=${retVal[1]}
queryDirs=`echo ${retVal[2]}|sed 's/,/ /g'`
queryFileName=${retVal[3]}
if [ "$indexDirs" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data. Data can be incomplete or inconsistent, e.g., it can mix new and old naming conventions."
  exit 1
else
  echo "Using the data input files: $dataFileName, $queryFileName"
fi

total_qty=0

#set -x

echo "Index dirs: ${indexDirs[*]}"
echo "Query dirs: ${queryDirs[*]}"

echo "Queries/questions:"
for part in ${queryDirs[*]}  ; do
  queryFilePath="$inputDataDir/$part/$queryFileName"

  # Not all parts correspond have a query file
  if [ -f "$queryFilePath" ] ; then

    wcq=(`wc "$queryFilePath"`)
    echo $part $wcq
  fi
done

echo "Documents/passages/answers:"
for part in ${indexDirs[*]}  ; do

  dataFilePath="$inputDataDir/$part/$dataFileName"

  # Not all parts have a data file
  if [ -f "$dataFilePath" ] ; then
    catCommand=`getCatCmd "$dataFilePath"`

    wca=(`"$catCommand" "$dataFilePath" |wc`)
    echo $part $wca
  fi

done



