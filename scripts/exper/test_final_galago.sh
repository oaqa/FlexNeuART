#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

QREL_TYPE=$2
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""

MAX_QUERY_QTY="$3"

NUM_CPU_CORES=$4

if [ "$NUM_CPU_CORES" = "" ] ; then
  NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
  check "getting the number of CPU cores, do you have /proc/cpu/info?"
fi

PARALLEL_EXPER_QTY=1
THREAD_QTY=$(($NUM_CPU_CORES/$PARALLEL_EXPER_QTY))

TEST_SET="test"
EXPER_DIR="results/final/${collect}/$QREL_FILE/$TEST_SET/galago/"

echo "The number of CPU cores:      $NUM_CPU_CORES"
echo "The number of || experiments: $PARALLEL_EXPER_QTY"
echo "The number of threads:        $THREAD_QTY"
echo "Max # of queries to use:      $MAX_QUERY_QTY"
echo "QREL file:                    $QREL_FILE"
echo "Experimentation directory:    $EXPER_DIR"

if [ ! -d "$EXPER_DIR" ] ; then
  mkdir -p $EXPER_DIR
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$EXPER_DIR'"
    exit 1
  fi
fi

. scripts/num_ret_list.sh

# 2 combine runs, the first one is the warm up run!
EXPER_DESC=()
EXPER_DESC+=("combine @") 
EXPER_DESC+=("combine @")

if [ "$collect" = "compr" ] ; then
  EXPER_DESC+=("sdm uniw=0.9,odw=0.1,uww=0.0" \
              "rm fbDocs=5,fbTerm=100,fbOrigWeight=0.9")
elif [ "$collect" = "stackoverflow" ] ; then
  EXPER_DESC+=("sdm uniw=0.8,odw=0.2,uww=0.0" \
              "rm fbDocs=5,fbTerm=100,fbOrigWeight=0.9")
else
  echo "Unsupported collection: $collect"
  exit 1
fi


n=${#EXPER_DESC[*]}
childPIDs=()
nrun=0
nfail=0

echo "Experimental descriptors:"
for ((i=0;i<$n;++i))
  do
    line=${EXPER_DESC[$i]}
    if [ "$line" !=  "" ] ; then
      echo $line
    fi
  done
echo "====================="


for ((i=0;i<$n;++i))
  do
    line=${EXPER_DESC[$i]}
    if [ "$line" !=  "" ]
    then
      echo "Description line: $line"
      GALAGO_OP=`echo $line|awk '{print $1}'`
      if [ "$GALAGO_OP" = "" ] ; then
        echo "Missing Galago operator type (1st field) in line $line, file $EXPER_DESC_FILE"
        exit 1
      fi
      GALAGO_PARAMS=`echo $line|awk '{print $2}'`
      if [ "$GALAGO_PARAMS" = "" ] ; then
        echo "Missing param list (2d field) in line $line, file $EXPER_DESC_FILE, use @ to indicate empty list"
        exit 1
      fi
      # Each experiment should run in its separate directory
      EXPER_DIR_UNIQUE="$EXPER_DIR/$GALAGO_OP/$GALAGO_PARAMS"
      if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
        mkdir -p "$EXPER_DIR_UNIQUE"
        if [ "$?" != "0" ] ; then
          echo "Failed to create $EXPER_DIR_UNIQUE"
          exit 1
        fi
      fi
      # Galago experiments currently don't use training
      scripts/exper/run_one_galago_exper.sh $collect "$QREL_FILE" "$EXPER_DIR_UNIQUE" "$GALAGO_OP" "$GALAGO_PARAMS" "$MAX_QUERY_QTY"  "$TEST_SET" "$THREAD_QTY" "$NUM_RET_LIST" &> $EXPER_DIR_UNIQUE/exper.log &
      pid=$!
      childPIDs+=($pid)
      echo "Started a process $pid, working dir: $EXPER_DIR_UNIQUE"
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
echo "============================================"

