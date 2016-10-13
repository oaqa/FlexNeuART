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

PID=""

function start_server {
  NMSLIB_INDEX=$1
  PROG_NAME="query_server"

  #if [ ! -f "$NMSLIB_INDEX" ] ; then
    #echo "Can't find index file: $NMSLIB_INDEX"
    #exit 1
  #fi

  # For simplicity, we assume that there would be only one instance of the query server running on the experimentation server!
  pgrep $PROG_NAME &> /dev/null
  if [ "$?" = "0" ] ; then
    echo "It looks like one instance of $PROG_NAME is already running!"
    exit 1
  fi

  $NMSLIB_PATH_SERVER/query_server -s $NMSLIB_SPACE -i $NMSLIB_HEADER -p $NMSLIB_PORT -m $NMSLIB_METHOD -L $NMSLIB_INDEX -S $NMSLIB_INDEX &> server.log  &

  PID=$!

  echo $PID > server.pid
  check "echo $PID > server.pid"

  # Now we will keep checking if the server started

  started=0
  while [ "$started" = "0" ] 
  do
    sleep 10
    echo "Checking if NMSLIB server (PID=$PID) has started"
    ps -p $PID &>/dev/null
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

collect="compr"
echo "Collection: $collect"
#collect=${POS_ARGS[0]}
#if [ "$collect" = "" ] ; then
  #echo "Specify a collection: manner, compr (1st arg)"
  #exit 1
#fi

TEST_PART=${POS_ARGS[0]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (1st arg)"
  exit 1 
fi

NMSLIB_INDEX_DIR="${POS_ARGS[1]}"

if [ "$NMSLIB_INDEX_DIR" = "" ] ; then
  echo "Specify a location of the directory with NAPP indices (2d positional arg)!"
  exit 1
fi

if [ ! -d "$NMSLIB_INDEX_DIR" ] ; then
  echo "$NMSLIB_INDEX_DIR is not a directory (2rd positional arg)!"
  exit 1
fi

NMSLIB_HEADER_NAME="header_bm25_text"
EXPER_DIR_BASE=results/final/$collect/$TEST_PART/nmslib/napp/$NMSLIB_HEADER_NAME

echo "Header: $NMSLIB_HEADER_NAME"
echo "Base exper dir: $EXPER_DIR_BASE"

CAND_PROV_TYPE="nmslib"
NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
#EXTR_TYPE_FINAL="complex"
#EXTR_MODEL_FINAL="results/final/$collect/train/complex/exper/out_${collect}_train_complex_50.model"
EXTR_TYPE_FINAL="none"
EXTR_MODEL_FINAL="none"
NMSLIB_SPACE="qa1"
NMSLIB_METHOD="napp_qa1"
NMSLIB_FIELDS="text"
NMSLIB_PORT=10000
NMSLIB_HEADER="nmslib/$collect/headers/$NMSLIB_HEADER_NAME"
NMSLIB_PATH_SERVER=../nmslib/query_server/cpp_client_server
WORD_EMBEDDINGS="word2vec_retro_unweighted_minProb=0.001.txt"

echo "The number of threads:       $THREAD_QTY"
if [ "$max_num_query_param" != "" ] ; then
  echo "Max # of queries param:      $max_num_query_param"
fi


PARAMS=( \
"napp_numPivot=8000,numPivotIndex=238_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=12" \
"napp_numPivot=8000,numPivotIndex=228_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=13" \
\
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=11" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=12" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=13" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=14" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=15" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=17" \
"napp_numPivot=8000,numPivotIndex=200_pivots_text_field_maxTermQty50K_pivotTermQty1000" "numPivotSearch=19" \
)



# Now let's loop over the list of query-time & index-time parameters and carry out an experiment for each setting. 

PREV_INDEX=""

pnum=$((${#PARAMS[*]}/2))
for ((i=0;i<$pnum;++i))
do
  ii=$((2*$i))
  iq=$((2*$i+1))

  index_name=${PARAMS[$ii]}
  query_time_params=${PARAMS[$iq]}

  echo "Index name: $index_name"
  echo "Query time parameters: $query_time_params"

  # Start server only if the index has changed
  if [ "$PREV_INDEX" != "$index_name" ] ; then
    if [ "$PREV_INDEX" != "" ] ; then
      echo "Trying to kill the server with PID=$PID"
      kill -9 $PID ; check "kill -9 $PID"
      # Dying takes some time
      sleep 10
      kill -9 $PID 
      sleep 5
    fi
    start_server $NMSLIB_INDEX_DIR/$index_name
    PREV_INDEX=$index_name
  fi

  $NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params 
  check "$NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params "
  echo "Successfully set NMSLIB server parameters: $query_time_params"

  EXPER_DIR=$EXPER_DIR_BASE/$index_name/$query_time_params
  mkdir -p $EXPER_DIR
  check "mkdir -p $EXPER_DIR"
  cmd="scripts/exper/test_final_model.sh $collect $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param "
  bash -c "$cmd"
  check "$cmd"

done

# In the end, stop the query_server
kill -9 $PID ; check "kill -9 $PID"

