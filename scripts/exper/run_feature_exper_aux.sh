#!/bin/bash
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr"
  exit 1
fi

QREL_FILE=$2
if [ "$QREL_FILE" = "" ] ; then
  echo "Specify QREL file (2rd arg)"
  exit 1
fi

EXPER_DIR=$3
if [ "$EXPER_DIR" = "" ] ; then
  echo "Specify a working directory prefix(3d arg)!"
  exit 1
fi
if [ ! -d "$EXPER_DIR" ] ; then
  mkdir -p $EXPER_DIR
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$EXPER_DIR'"
    exit 1
  fi
fi

if [ "$collect" = "manner" ] ; then
  NUM_RET_LIST="10,15,17,36,72"
  N_TRAIN=15
elif [ "$collect" = "compr" ] ; then
  #NUM_RET_LIST="50,100,200,400,600,1000,1500,2500"
  # Not much value beoynd 400
  #NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,100,200,400"
  NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
  N_TRAIN=15
elif [ "$collect" = "stackoverflow" ] ; then
  NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
  N_TRAIN=15
else
  echo "Unsupported collection: $collect"
  exit 1
fi

EXTRACTORS_DESC=$4
if [ "$EXTRACTORS_DESC" = "" ] ; then
  "Specify a file with extractor description (4th arg)"
  exit 1
fi
if [ ! -f "$EXTRACTORS_DESC" ] ; then
  "Not a file '$EXTRACTORS_DESC' (4th arg)"
  exit 1
fi

PARALLEL_EXPER_QTY=$5
if [ "$PARALLEL_EXPER_QTY" = "" ] ; then
  echo "Specify a number of experiments that are run in parallel (5th arg)!"
  exit 1
fi

THREAD_QTY=$6
if [ "$THREAD_QTY" = "" ] ; then
  echo "Specify a number of threads for the feature extractor (6th arg)!"
  exit 1
fi

MAX_QUERY_QTY="$7"

nTotal=0
nRunning=0


echo "Number of parallel experiments:                             $PARALLEL_EXPER_QTY"
echo "Number of threads in feature extractors/query applications: $THREAD_QTY"

function wait_children {
  echo "Waiting for ${#childPIDs[*]} child processes"
  for pid in ${childPIDs[*]} ; do
    wait $pid
    stat=$?
    if [ "$stat" != "0" ] ; then
      echo "Process with pid=$pid *FAILED*, status=$stat!"
      nfail=$(($nfail+1))
    else
      echo "Process with pid=$pid finished successfully."
    fi
  done
  childPIDs=()
}

n=`wc -l "$EXTRACTORS_DESC"|awk '{print $1}'`
n=$(($n+1))
childPIDs=()
nrun=0
nfail=0
for ((i=1;i<$n;++i))
  do
    line=`head -$i "$EXTRACTORS_DESC"|tail -1`
    if [ "$line" !=  "" ]
    then
      EXTR_TYPE=`echo $line|awk '{print $1}'`
      if [ "$EXTR_TYPE" = "" ] ; then
        echo "Missing extractor type (1st field) in line $line, file $EXTRACTORS_DESC"
        exit 1
      fi
      EMBED_LIST=`echo $line|awk '{print $2}'`
      if [ "$EMBED_LIST" = "" ] ; then
        echo "Missing embedding list (2d field) in line $line, file $EXTRACTORS_DESC"
        exit 1
      fi
      TEST_SET=`echo $line|awk '{print $3}'`
      if [ "$TEST_SET" = "" ] ; then
        echo "Missing test set (e.g., dev1) (3d field) in line $line, file $EXTRACTORS_DESC"
        exit 1
      fi
      # Each experiment should run in its separate directory
      EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$QREL_FILE/$TEST_SET/$EXTR_TYPE/$EMBED_LIST"
      if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
        mkdir -p "$EXPER_DIR_UNIQUE"
        if [ "$?" != "0" ] ; then
          echo "Failed to create $EXPER_DIR_UNIQUE"
          exit 1
        fi
      fi
      scripts/exper/run_one_exper.sh $collect "$QREL_FILE" "$EXPER_DIR_UNIQUE" "$EXTR_TYPE" "$MAX_QUERY_QTY"  "$TEST_SET" "$THREAD_QTY" "$NUM_RET_LIST" "$N_TRAIN" "$EMBED_LIST" &> $EXPER_DIR_UNIQUE/exper.log &
      pid=$!
      childPIDs[$nrun]=$pid
      echo "Started a process $pid, working dir: $EXPER_DIR_UNIQUE"
      nRunning=$(($nRunning+1))
      nrun=$(($nrun+1))
    fi
    if [ "$nRunning" -ge $PARALLEL_EXPER_QTY ] ; then
      wait_children
      nRunning=0
    fi
  done
wait_children
echo "============================================"
echo "$nrun experiments executed"
echo "$nfail experiments failed"
echo "============================================"

