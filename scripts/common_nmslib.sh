
NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
check "getting the number of CPU cores, do you have /proc/cpu/info?"

THREAD_QTY=$NUM_CPU_CORES
max_num_query_param=""
max_num_query=""

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
        max_num_query=$OPT_VALUE
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

PID=""

function start_server {
  NMSLIB_INDEX=$1
  INDEX_PARAMS=$2
  PROG_NAME="query_server"

  NMSLIB_INDEX_COMP="${NMSLIB_INDEX}.gz"

  if [ -f "$NMSLIB_INDEX_COMP" ] ; then
    echo "Let's uncompress previously created index $NMSLIB_INDEX_COMP"
    zcat "$NMSLIB_INDEX_COMP" > "$NMSLIB_INDEX"
    check "zcat \"$NMSLIB_INDEX_COMP\" > \"$NMSLIB_INDEX\""
  fi

  if [ ! -f "$NMSLIB_INDEX" ] ; then
    echo "Warning: Can't find index file: $NMSLIB_INDEX or its compressed variant $NMSLIB_INDEX_COMP"
  fi

  # We assume that there would be only one instance of the query server running on the experimentation server!
  pgrep $PROG_NAME &> /dev/null
  if [ "$?" = "0" ] ; then
    echo "It looks like one instance of $PROG_NAME is already running!"
    exit 1
  fi

  $NMSLIB_PATH_SERVER/query_server -s $NMSLIB_SPACE -i $NMSLIB_HEADER -p $NMSLIB_PORT -m $NMSLIB_METHOD -c $INDEX_PARAMS -L $NMSLIB_INDEX -S $NMSLIB_INDEX &> $SERVER_LOG_NAME  &

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
    tail -1 $SERVER_LOG_NAME | grep 'Started a server' &>/dev/null
    if [ "$?" = "0" ] ; then
      echo "NMSLIB query server has started!"
      started=1
    fi
  done
}


