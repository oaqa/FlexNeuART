# Some common function to share

SERVER_LOG_NAME="server.log"

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

function wait_children {
  pidLIST=($@)
  echo "Waiting for ${#pidLIST[*]} child processes"
  for pid in ${pidLIST[*]} ; do
    wait $pid
    stat=$?
    if [ "$stat" != "0" ] ; then
      echo "Process with pid=$pid *FAILED*, status=$stat!"
      nfail=$(($nfail+1))
    else
      echo "Process with pid=$pid finished successfully."
    fi
  done
}

function get_qrel_file {
  QREL_TYPE=$1
  ARG_NUM=$2
  QREL_FILE=""

  if [ "$QREL_TYPE" = "graded" ] ; then
    QREL_FILE="qrels_all_graded.txt"
  elif [ "$QREL_TYPE" = "binary" ] ; then
    QREL_FILE="qrels_all_binary.txt"
  elif [ "$QREL_TYPE" = "onlybest" ] ; then
    QREL_FILE="qrels_onlybest.txt"
  elif [ "$QREL_TYPE" = "graded_same_score" ] ; then
    QREL_FILE="qrels_all_graded_same_score.txt"
  elif [ "$QREL_TYPE" = "" ] ; then
    echo "Specifiy QREL type ($ARG_NUM arg)" 1>&2
    exit 1
  else
    echo "Unsupported QREL type ($ARG_NUM arg) $QREL_TYPE, expected binary, onlybest, graded, graded_same_score" 1>&2
    exit 1
  fi
  if [ "$QREL_FILE" = "" ] ; then
    echo "Bug QREL_FILE is empty for QREL_TYPE=$QREL_TYPE" 1>&2
    exit 1
  fi
  echo $QREL_FILE
}

function save_server_logs {
  me=`basename "$0"`
  mv $SERVER_LOG_NAME $SERVER_LOG_NAME.$me
}

function setJavaMem {
  echo "HEEERE!"
  F1="$1"
  F2="$2"
  OS=`uname|awk '{print $1}'`
  if [ "$OS" = "Linux" ] ; then
    MEM_SIZE_MX_KB=`free|grep Mem|awk '{print $2}'`
  elif [ "$OS" = "Darwin" ] ; then
    # Assuming Macbook pro
    MEM_SIZE_MX_KB=$((16384*1024))
  else
    echo "Unsupported OS: $OS"
    exit 1
  fi
  # No mx
  MEM_SIZE_MIN_KB=$(($F1*$MEM_SIZE_MX_KB/$F2))
  MEM_SIZE_MAX_KB=$((7*$MEM_SIZE_MX_KB/8))
  export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MAX_KB}k -server"
  echo "MAVEN_OPTS=$MAVEN_OPTS"
}
