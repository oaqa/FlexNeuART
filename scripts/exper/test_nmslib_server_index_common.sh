
EXPER_DIR_BASE=results/final/$collect/$QREL_FILE/$TEST_PART/nmslib/$INDEX_METHOD_PREFIX/$NMSLIB_HEADER_NAME

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

prev_index_name=""

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
  if [ "$prev_index_name" != "$index_name" ] ; then
    if [ "$prev_index_name" != "" ] ; then
      echo "Trying to kill the server with PID=$PID"
      kill -9 $PID ; check "kill -9 $PID"
      # Dying takes some time
      sleep 10
      kill -9 $PID 
      sleep 15
    fi
    NMSLIB_INDEX="$NMSLIB_INDEX_DIR/$index_name"
    if [ "$PREV_INDEX_COMP" != "" -a -f "$PREV_INDEX_COMP" ] ; then
      echo "Removing uncompressed index $PREV_INDEX, because compressed exists!"
      rm "$PREV_INDEX"
      check "rm $PREV_INDEX"
    fi
    start_server "$NMSLIB_INDEX" $index_params
    prev_index_name=$index_name
    PREV_INDEX=$NMSLIB_INDEX
    PREV_INDEX_COMP="${NMSLIB_INDEX}.gz"
  fi

  $NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params 
  check "$NMSLIB_PATH_SERVER/query_client -a localhost -p $NMSLIB_PORT -t $query_time_params "
  echo "Successfully set NMSLIB server parameters: $query_time_params"

  EXPER_DIR=$EXPER_DIR_BASE/$index_name/$query_time_params
  mkdir -p $EXPER_DIR
  check "mkdir -p $EXPER_DIR"
  cmd="scripts/exper/test_final_model.sh $collect $QREL_FILE $TEST_PART nmslib -nmslib_addr localhost:$NMSLIB_PORT -nmslib_fields $NMSLIB_FIELDS "$EXPER_DIR" $EXTR_TYPE_FINAL $EXTR_MODEL_FINAL $NUM_RET_LIST $WORD_EMBEDDINGS -thread_qty $THREAD_QTY $max_num_query_param  -dont_delete_trec_runs "
  bash -c "$cmd"
  check "$cmd"

done

if [ "$PREV_INDEX_COMP" != "" -a -f "$PREV_INDEX_COMP" ] ; then
  echo "Removing uncompressed index $PREV_INDEX, because compressed exists!"
  rm "$PREV_INDEX"
  check "rm $PREV_INDEX"
fi

# In the end, stop the query_server
kill -9 $PID ; check "kill -9 $PID"
sleep 15

save_server_logs
