#!/bin/bash

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

function check_pipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
    exit 1
  fi
}


NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
check "getting the number of CPU cores, do you have /proc/cpu/info?"

THREAD_QTY=$NUM_CPU_CORES
max_num_query_param=""

while [ $# -ne 0 ] ; do
  echo $1|grep "^-" >/dev/null 
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    OPT_VALUE="$2"
    OPT="$1 $2"
    if [ "$OPT_VALUE" = "" ] ; then  
      echo "Option $OPT_NAME requires an argument." >&2
      exit 1
    fi
    shift 2
    case $OPT_NAME in
      -thread_qty)
        THREAD_QTY=$OPT_VALUE 
        ;;
      -max_num_query)
        max_num_query_param=$OPT
        ;;
      *)
        echo "Invalid option: $OPT_NAME" >&2
        exit 1
        ;;
    esac
  else
    POS_ARGS=(${POS_ARGS[*]} $1)
    shift 1
  fi
done

collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

TEST_PART=${POS_ARGS[1]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (2d arg)"
  exit 1 
fi

HEADER_NAME="header_bm25_text"
EXPER_DIR_BASE="results/final/$collect/$TEST_PART/nmslib/brute_force/$HEADER_NAME"


CAND_PROV_TYPE="nmslib"
NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
#EXTR_TYPE_FINAL="complex"
#EXTR_MODEL_FINAL="results/final/$collect/train/complex/exper/out_${collect}_train_complex_50.model"
EXTR_TYPE_FINAL="none"
EXTR_MODEL_FINAL="none"
NMSLIB_SPACE="qa1"
NMSLIB_METHOD="brute_force"
NMSLIB_FIELDS="text"
NMSLIB_PORT=10000
NMSLIB_HEADER="nmslib/$collect/headers/$HEADER_NAME"
NMSLIB_PATH_SERVER=../nmslib4qa/query_server/cpp_client_server
WORD_EMBEDDINGS="word2vec_retro_unweighted_minProb=0.001.txt"

echo "The number of threads:       $THREAD_QTY"
if [ "$max_num_query_param" != "" ] ; then
  echo "Max # of queries param:      $max_num_query_param"
fi


PROG_NAME="query_server"

# For simplicity, we assume that there would be only one instance of the query server running on the experimentation server!
pgrep $PROG_NAME &> /dev/null
if [ "$?" = "0" ] ; then
  echo "It looks like one instance of $PROG_NAME is already running!"
  exit 1
fi

$NMSLIB_PATH_SERVER/query_server -s $NMSLIB_SPACE -i $NMSLIB_HEADER -p $NMSLIB_PORT -m $NMSLIB_METHOD &> server.log  &

pid=$!

echo $pid > server.pid
check "echo $pid > server.pid"

# Now we will keep checking if the server started

started=0
while [ "$started" = "0" ] 
do
  sleep 10
  echo "Checking if NMSLIB server (PID=$pid) has started"
  ps -p $pid &>/dev/null
  if [ "${PIPESTATUS[0]}" != "0" ] ; then
    echo "NMSLIB query server stopped unexpectedly, check logs"
    exit 1
  fi
  tail -1 server.log | grep 'Started a server' &>/dev/null
  if [ "$?" = "0" ] ; then
    echo "NMSLIB query server has started!"
    started=1
  fi
done

EXPER_DIR=$EXPER_DIR_BASE
mkdir -p $EXPER_DIR
check "mkdir -p $EXPER_DIR"

# Note that we don't delete trec_runs, can be used later to evaluate effectiveness of NAPP from the Java program!
cmd="scripts/exper/test_final_model.sh $collect $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param -dont_delete_trec_runs "
bash -c "$cmd"
check "$cmd"

# In the end, stop the query_server

kill -9 $pid
check "kill -9 $pid"
