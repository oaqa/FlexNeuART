#!/bin/bash -e
source scripts/config.sh
source scripts/common_proc.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "QREL_FILE"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"

oldCollPref=$1;
if [ "$oldCollPref" = "" ] ; then
  echo "Specify the prefix name of the current collection (1st arg)"
  exit 1
fi

newCollPref=$2;
if [ "$oldCollPref" = "" ] ; then
  echo "Specify the prefix name of the new splitted collection (2nd arg)"
  exit 1
fi

for colType in nq trivia squad ; do
  trainFusionSize=10000;
  if [ "$colType" = "squad" ] ; then
    trainFusionSize=5000;
  fi

  oldCollect="${oldCollPref}_${colType}"
  newCollect="${newCollPref}_${colType}"

  oldCollectDir="$COLLECT_ROOT/$oldCollect/$INPUT_DATA_SUBDIR"
  newCollectDir="$COLLECT_ROOT/$newCollect/$INPUT_DATA_SUBDIR"

  mkdir -p $newCollectDir

  cp -r ${oldCollectDir}/dev ${newCollectDir}/dev_official
  cp -r ${oldCollectDir}/pass ${newCollectDir}/pass

  python -u scripts/data_convert/wikipedia_dpr/split_queries.py \
                --src_dir "${oldCollectDir}/train" \
                --dst_dir "$newCollectDir" \
                --partitions_sizes="-1,5000,5000" \
                --partitions_names "bitext,train_fusion,dev"
done