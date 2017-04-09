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

THREAD_QTY=$NUM_CPU_CORES

TEST_SET="test"
EXPER_DIR="results/final/${collect}/$QREL_FILE/$TEST_SET/galago/"

DO_WARMUP="0"

echo "The number of CPU cores:      $NUM_CPU_CORES"
echo "The number of threads:        $THREAD_QTY"
echo "Max # of queries to use:      $MAX_QUERY_QTY"
echo "QREL file:                    $QREL_FILE"
echo "Experimentation directory:    $EXPER_DIR"
echo "Warm-up run?:                 $DO_WARMUP"

if [ ! -d "$EXPER_DIR" ] ; then
  mkdir -p $EXPER_DIR
  if [ "$?" != "0" ] ; then
    echo "Cannot create '$EXPER_DIR'"
    exit 1
  fi
fi

. scripts/num_ret_list.sh

# 2 combine runs
EXPER_DESC=("combine @") 

if [ "$collect" = "compr" ] ; then
  EXPER_DESC+=("sdm uniw=0.9,odw=0.1,uww=0.0" \
              "rm fbDocs=5,fbTerm=100,fbOrigWeight=0.9")
elif [ "$collect" = "stackoverflow" ] ; then
  EXPER_DESC+=("sdm uniw=0.8,odw=0.2,uww=0.0" \
              "rm fbDocs=5,fbTerm=100,fbOrigWeight=0.9")
elif [ "$collect" = "squad" ] ; then
  EXPER_DESC+=("sdm scorer=default,uniw=0.8,odw=0.15,uww=0.1" \
              "sdm scorer" \
              "sdm scorer=default @" \
              "rm scorer=default,fbOrigWeight=0.75,fbDocs=5,fbTerm=140" \
              )
else
  echo "Unsupported collection: $collect"
  exit 1
fi


n=${#EXPER_DESC[*]}
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
      if [ "$DO_WARMUP" = "1" ] ; then
        # Galago experiments currently don't use training: the first run is a warm-up!
        echo "Warmup run"
        scripts/exper/run_one_galago_exper.sh $collect "$QREL_FILE" "$EXPER_DIR_UNIQUE" "$GALAGO_OP" "$GALAGO_PARAMS" "$MAX_QUERY_QTY"  "$TEST_SET" "$THREAD_QTY" "$NUM_RET_LIST" &> $EXPER_DIR_UNIQUE/exper.log 
        check "run_one_galago_exper.sh $collect ... "
        echo "Real test run"
      fi
      scripts/exper/run_one_galago_exper.sh $collect "$QREL_FILE" "$EXPER_DIR_UNIQUE" "$GALAGO_OP" "$GALAGO_PARAMS" "$MAX_QUERY_QTY"  "$TEST_SET" "$THREAD_QTY" "$NUM_RET_LIST" &> $EXPER_DIR_UNIQUE/exper.log 
    fi
      check "run_one_galago_exper.sh $collect ... "
  done
echo "============================================"
echo "$nrun experiments executed"
echo "$nfail experiments failed"
echo "============================================"

