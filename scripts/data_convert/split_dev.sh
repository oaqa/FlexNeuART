#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "DEV_SUBDIR"
checkVarNonEmpty "DEV1_SUBDIR"
checkVarNonEmpty "DEV2_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"

dev1qty=$2

if [ "$dev1qty" = "" ] ; then
  echo "Specify a number of dev1 queries (2d arg)"
  exit 1
fi

tmpFileQueries=`mktemp`

inputDataDir="$COLLECT_ROOT/$collect/$INPUT_DATA_SUBDIR"

echo "=========================================================================="
echo "Data directory:          $inputDataDir"

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

echo "Query file name:         $queryFileName"
echo "=========================================================================="

for devPart in "$DEV1_SUBDIR" "$DEV2_SUBDIR" ; do
  if [ ! -d "$inputDataDir/$devPart" ] ; then
    mkdir "$inputDataDir/$devPart"
  fi
  # Simply replicate the qrel file from dev it's not too big
  # to worry about splitting it
  cp "$inputDataDir/dev/$QREL_FILE" "$inputDataDir/$devPart/"
done

fullQueryPath=$inputDataDir/$DEV_SUBDIR/$queryFileName
qty=`wc -l "$fullQueryPath"|awk '{print $1}'` 
dev2qty=$(($qty-$dev1qty))
echo "# of queries: $qty # of queries in "$DEV1_SUBDIR": $dev1qty # of queries in $DEV2_SUBDIR: $dev2qty"
if [ "$dev2qty" -lt "1" ] ; then
  echo "Requested number of "$DEV1_SUBDIR" queries $dev1qty is too large!"
  exit 1
fi
execAndCheck "sort -R \"$fullQueryPath\" > \"$tmpFileQueries\""
execAndCheck "head -${dev1qty} \"$tmpFileQueries\" > \"$inputDataDir/dev1/$queryFileName\""
execAndCheck "tail -${dev2qty} \"$tmpFileQueries\" > \"$inputDataDir/dev2/$queryFileName\""

rm $tmpFileQueries
