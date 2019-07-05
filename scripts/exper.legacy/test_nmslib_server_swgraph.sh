#!/bin/bash
. scripts/common.sh
. scripts/common_nmslib.sh

collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, stackoverflow (1st arg)"
  exit 1
fi

QREL_TYPE=${POS_ARGS[1]}
QREL_FILE=`get_qrel_file $QREL_TYPE "2d"`
check ""

TEST_PART=${POS_ARGS[2]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (3d arg)"
  exit 1 
fi

NMSLIB_HEADER_NAME="header_avg_embed_word2vec_text_unlemm"

EXPER_DIR_BASE=results/final/$collect/$QREL_FILE/$TEST_PART/nmslib/sw-graph

NMSLIB_INDEX_DIR="nmslib/$collect/index/test/$NMSLIB_HEADER_NAME"
if [ ! -d "$NMSLIB_INDEX_DIR" ] ; then
  mkdir -p $NMSLIB_INDEX_DIR ; 
  check "mkdir -p $NMSLIB_INDEX_DIR"
fi

INDEX_PARAMS="NN=10,initIndexAttempts=2,efConstruction=900"

NMSLIB_INDEX="$NMSLIB_INDEX_DIR/sw-graph_${INDEX_PARAMS}"

echo "Header: $NMSLIB_HEADER_NAME"
echo "Base exper dir: $EXPER_DIR_BASE"
echo "NMSLIB index dir: $NMSLIB_INDEX_DIR"
echo "NMSLIB index: $NMSLIB_INDEX"

EXTR_TYPE_FINAL="none"
EXTR_MODEL_FINAL="none"


CAND_PROV_TYPE="nmslib"
CAND_QTY="500"
NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
#EXTR_TYPE_FINAL="complex"
#EXTR_MODEL_FINAL="results/final/$collect/train/complex/exper/out_${collect}_train_complex_50.model"
EXTR_TYPE_FINAL="none"
EXTR_MODEL_FINAL="none"
NMSLIB_SPACE="qa1"
NMSLIB_METHOD="sw-graph"
NMSLIB_FIELDS="text_unlemm"
NMSLIB_PORT=10000
NMSLIB_HEADER="nmslib/$collect/headers/$NMSLIB_HEADER_NAME"
NMSLIB_PATH_SERVER=../nmslib/query_server/cpp_client_server
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

$NMSLIB_PATH_SERVER/query_server -s $NMSLIB_SPACE -i $NMSLIB_HEADER -p $NMSLIB_PORT -m $NMSLIB_METHOD -c $INDEX_PARAMS -L $NMSLIB_INDEX -S $NMSLIB_INDEX &> $SERVER_LOG_NAME  &

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
  tail -1 $SERVER_LOG_NAME | grep 'Started a server' &>/dev/null
  if [ "$?" = "0" ] ; then
    echo "NMSLIB query server has started!"
    started=1
  fi
done

QUERY_TIME_PARAMS=(initSearchAttempts=1,efSearch=450  \
                   initSearchAttempts=1,efSearch=900  \
                   initSearchAttempts=1,efSearch=1800 \
                   initSearchAttempts=1,efSearch=2100 \
                   initSearchAttempts=1,efSearch=2400 \
                   initSearchAttempts=1,efSearch=3600 \
                   initSearchAttempts=1,efSearch=7200 )

# Now let's loop over the list of query-time parameters and carry out 
# an experiment for each setting. 

for param in ${QUERY_TIME_PARAMS[*]} 
do
  # Here we test a simpler and a more complex intermedaite re-ranking model
  $NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $param 
  check "$NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $param "
  echo "Successfully set NMSLIB server parameters: $param"

  EXPER_DIR=$EXPER_DIR_BASE/no_rerank/$param
  mkdir -p $EXPER_DIR
  check "mkdir -p $EXPER_DIR"
  cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param -dont_delete_trec_runs"
  bash -c "$cmd"
  check "$cmd"

  #EXPER_DIR=$EXPER_DIR_BASE/bm25_rerank/$param
  #mkdir -p $EXPER_DIR
  #check "mkdir -p $EXPER_DIR"
  #cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param -extr_type_interm exper@bm25=text -model_interm nmslib/${collect}/models/one_feature.model  -cand_qty $CAND_QTY -dont_delete_trec_runs"
  #bash -c "$cmd"
  #check "$cmd"

  #EXPER_DIR=$EXPER_DIR_BASE/basic_interm_rerank/$param
  #mkdir -p $EXPER_DIR
  #check "mkdir -p $EXPER_DIR"
  #cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param -extr_type_interm exper@bm25=text,text_unlemm,bigram+overall_match=text,text_unlemm,bigram -model_interm nmslib/${collect}/models/out_${collect}_train_exper@bm25=text,text_unlemm,bigram+overall_match=text,text_unlemm,bigram_15.model    -cand_qty $CAND_QTY -dont_delete_trec_runs"
  #bash -c "$cmd"
  #check "$cmd"

  #EXPER_DIR=$EXPER_DIR_BASE/tran_interm_rerank/$param
  #mkdir -p $EXPER_DIR
  #check "mkdir -p $EXPER_DIR"
  #cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param  -extr_type_interm exper@bm25=text,text_unlemm,bigram+model1=text+simple_tran=text -model_interm nmslib/${collect}/models/out_${collect}_train_exper@bm25=text,text_unlemm,bigram+model1=text+simple_tran=text_15.model  -cand_qty $CAND_QTY -dont_delete_trec_runs"
  #bash -c "$cmd"
  #check "$cmd"

done

# In the end, stop the query_server

kill -9 $pid
check "kill -9 $pid"
sleep 15

save_server_logs

