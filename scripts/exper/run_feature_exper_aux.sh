#!/bin/bash 
. scripts/common_proc.sh
. scripts/config.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection, e.g., squad (1st arg)"
  exit 1
fi


checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FEAT_EXPER_SUBDIR"
checkVarNonEmpty "EXPER_DESC_SUBDIR"

experDescLoc="$COLLECT_ROOT/$collect/$EXPER_DESC_SUBDIR"

. scripts/config_cand_qty.sh

EXTRACTORS_DESC=$2
if [ "$EXTRACTORS_DESC" = "" ] ; then
  "Specify a file with extractor description relative to dir. '$experDescLoc' (2d arg)"
  exit 1
fi
experDescPath=$experDescLoc/$EXTRACTORS_DESC
if [ ! -f "$experDescPath" ] ; then
  echo "Not a file '$experDescPath' (wrong 2d arg)"
  exit 1
fi

PARALLEL_EXPER_QTY=$3
if [ "$PARALLEL_EXPER_QTY" = "" ] ; then
  echo "Specify a number of experiments that are run in parallel (3d arg)!"
  exit 1
fi

THREAD_QTY=$4
if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify a number of threads for the feature extractor (4th arg)!"
  exit 1
fi

MAX_QUERY_QTY="$5"


experDir="$COLLECT_ROOT/$collect/$FEAT_EXPER_SUBDIR"
if [ ! -d "$experDir" ] ; then
  mkdir -p $experDir
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$experDir'"
    exit 1
  fi
fi

nTotal=0
nRunning=0


echo "Number of parallel experiments:                             $PARALLEL_EXPER_QTY"
echo "Number of threads in feature extractors/query applications: $THREAD_QTY"

n=`wc -l "$experDescPath"|awk '{print $1}'`
n=$(($n+1))
childPIDs=()
nrun=0
nfail=0
for ((ivar=1;ivar<$n;++ivar))
  do
    line=$(head -$ivar "$experDescPath"|tail -1)
    line=$(removeComment "$line")
    if [ "$line" !=  "" ]
    then
      extrConfigFile=`echo $line|awk '{print $1}'`
      if [ "$extrConfigFile" = "" ] ; then
        echo "Missing feature-extractor config file (1st field) in line $line, file $experDescPath"
        exit 1
      fi
      extrConfigPath="$experDescLoc/$extrConfigFile"
      if [ ! -f "$extrConfigPath" ] ; then
        echo "Missing feature-extractor configuration file: $extrConfigPath"
        exit 1
      fi
      testSet=`echo $line|awk '{print $2}'`
      if [ "$testSet" = "" ] ; then
        echo "Missing test set (e.g., dev1) (2d field) in line $line, file $experDescPath"
        exit 1
      fi
      experSubdir=`echo $line|awk '{print $3}'`
      if [ "$testSet" = "" ] ; then
        echo "Missing experimental sub-dir (3d field) in line $line, file $experDescPath"
        exit 1
      fi
      # Each experiment should run in its separate sub-directory
      experDirUnique=$(getExperDirUnique "$experDir" "$testSet" "$experSubdir")
      if [ ! -d "$experDirUnique" ] ; then
        mkdir -p "$experDirUnique"
        if [ "$?" != "0" ] ; then
          echo "Failed to create $experDirUnique"
          exit 1
        fi
      fi

      scripts/exper/run_one_lucene_exper.sh \
      $collect "$experDirUnique" \
      "$extrConfigPath" \
      "$MAX_QUERY_QTY"  \
      "$testSet" \
      "$THREAD_QTY" \
      "$NUM_RET_LIST" \
      "$N_TRAIN"  &> $experDirUnique/exper.log &

      pid=$!
      childPIDs+=($pid)
      echo "Started a process $pid, working dir: $experDirUnique"
      nRunning=$(($nRunning+1))
      nrun=$(($nrun+1))
    fi
    if [ "$nRunning" -ge $PARALLEL_EXPER_QTY ] ; then
      wait_children ${childPIDs[*]}
      childPIDs=()
      nRunning=0
    fi
  done
wait_children ${childPIDs[*]}
echo "============================================"
echo "$nrun experiments executed"
echo "$nfail experiments failed"
if [ "$nfail" -gt "0" ] ; then
  echo "Check the log in working directories!!!"
fi
echo "============================================"

