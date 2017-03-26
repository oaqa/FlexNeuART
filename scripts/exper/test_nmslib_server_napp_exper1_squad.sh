#!/bin/bash
. scripts/common.sh
. scripts/common_nmslib.sh

collect="squad"
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

QREL_TYPE=${POS_ARGS[1]}
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""

. scripts/exper/params/napp_exper1_squad.sh




EXPER_DIR_BASE=results/final/$collect/$QREL_FILE/$TEST_PART/nmslib/napp/$NMSLIB_HEADER_NAME

NMSLIB_INDEX_DIR="nmslib/$collect/index/test/$NMSLIB_HEADER_NAME"
if [ ! -d "$NMSLIB_INDEX_DIR" ] ; then
  mkdir -p $NMSLIB_INDEX_DIR ; 
  check "mkdir -p $NMSLIB_INDEX_DIR"
fi

echo "Header: $NMSLIB_HEADER_NAME"
echo "Base exper dir: $EXPER_DIR_BASE"
echo "NMSLIB index dir: $NMSLIB_INDEX_DIR"

CAND_PROV_TYPE="nmslib"
NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
#EXTR_TYPE_FINAL="complex"
#EXTR_MODEL_FINAL="results/final/$collect/train/complex/exper/out_${collect}_train_complex_50.model"
EXTR_TYPE_FINAL="none"
EXTR_MODEL_FINAL="none"
NMSLIB_SPACE="qa1"
NMSLIB_PORT=10000
NMSLIB_HEADER="nmslib/$collect/headers/$NMSLIB_HEADER_NAME"
NMSLIB_PATH_SERVER=../nmslib/query_server/cpp_client_server
WORD_EMBEDDINGS="word2vec_retro_unweighted_minProb=0.001.txt"

echo "The number of threads:       $THREAD_QTY"
if [ "$max_num_query_param" != "" ] ; then
  echo "Max # of queries param:      $max_num_query_param"
fi

# Now let's loop over the list of query-time & index-time parameters and carry out an experiment for each setting. 
PREV_INDEX=""

pnum=$((${#PARAMS[*]}/2))
for ((i=0;i<$pnum;++i))
do
  ii=$((2*$i))
  iq=$((2*$i+1))

  index_params=${PARAMS[$ii]}
  index_params_noslash=`echo $index_params|sed 's|/|_|g'`
  index_name=${INDEX_METHOD_PREFIX}_${index_params_noslash}
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
      sleep 15
    fi
    start_server $NMSLIB_INDEX_DIR/$index_name $index_params
    PREV_INDEX=$index_name
  fi

  $NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params 
  check "$NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params "
  echo "Successfully set NMSLIB server parameters: $query_time_params"

  EXPER_DIR=$EXPER_DIR_BASE/$index_name/$query_time_params
  mkdir -p $EXPER_DIR
  check "mkdir -p $EXPER_DIR"
  cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param "
  bash -c "$cmd"
  check "$cmd"

done

# In the end, stop the query_server
kill -9 $PID ; check "kill -9 $PID"
sleep 15

save_server_logs
