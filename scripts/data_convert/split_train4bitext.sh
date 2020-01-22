#!/bin/bash -e
. scripts/common_proc.sh
. scripts/config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "BITEXT_TRAIN_SUBDIR"
checkVarNonEmpty "DEFAULT_TRAIN_SUBDIR"

smallTrainQty=$2

if [ "$smallTrainQty" = "" ] ; then
  echo "Specify a number of train queries that should NOT be included into bi-text part (2d arg)"
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

if [ ! -d "$inputDataDir/$BITEXT_TRAIN_SUBDIR" ] ; then
  mkdir "$inputDataDir/$BITEXT_TRAIN_SUBDIR"
fi

fullQueryPath=$inputDataDir/$DEFAULT_TRAIN_SUBDIR/$queryFileName
cp "$inputDataDir/$DEFAULT_TRAIN_SUBDIR/$QREL_FILE" "$inputDataDir/$BITEXT_TRAIN_SUBDIR"
qty=`wc -l "$fullQueryPath"|awk '{print $1}'` 
qtyBitext=$(($qty-$smallTrainQty))
echo "# of queries: $qty # of queries for bi-text part: $qtyBitext"
if [ "$qtyBitext" -lt "1" ] ; then
  echo "Requested number of queries $smallTrainQty is too large!"
  exit 1
fi
execAndCheck "sort -R \"$fullQueryPath\" > \"$tmpFileQueries\""
execAndCheck "head -${smallTrainQty} \"$tmpFileQueries\" > \"$fullQueryPath\""
execAndCheck "tail -${qtyBitext} \"$tmpFileQueries\" > \"$inputDataDir/$BITEXT_TRAIN_SUBDIR/$queryFileName\""

rm $tmpFileQueries
