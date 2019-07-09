# Some common function to share

SERVER_LOG_NAME="server.log"

function execAndCheck {
  cmd0=$1
  desc=$2
  cmd="$cmd0"' ; (echo ${PIPESTATUS[*]} | grep -E "^0(\s+0)*$")'
  echo "$cmd0"
  bash -c "$cmd"
  # The status of the command sequence is
  #   i) The status of the last command, i.e., that of a grepping.
  #   ii)  *OR* the failure status of the whole bash -c operation,
  #        if it fails for some reason. One common reason: syntax error.
  if [ "$?" != "0" ] ; then
      echo "********************************************************************************"
      if [ "$desc" != "" ] ; then
        echo "  Command $desc failed:"
      else
        echo "  Command failed:"
      fi
      echo "$cmd0"
      echo "  Expanded cmd that was actually run in a seperate shell:"
      echo "$cmd"
      echo "********************************************************************************"
      exit 1
  fi
}


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

function checkVarNonEmpty {
  name="$1"
  val="${!name}"
  if [ "$val" = "" ] ; then
    echo "Variable $name is not set!"
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

function getOS {
  uname|awk '{print $1}'
}

function setJavaMem {
  F1="$1"
  F2="$2"
  OS=$(getOS)
  if [ "$OS" = "Linux" ] ; then
    MEM_SIZE_MX_KB=`free|grep Mem|awk '{print $2}'`
  elif [ "$OS" = "Darwin" ] ; then
    # Assuming Macbook pro
    MEM_SIZE_MX_KB=$((16384*1024))
  else
    echo "Unsupported OS: $OS" 1>&2
    exit 1
  fi
  # No mx
  MEM_SIZE_MIN_KB=$(($F1*$MEM_SIZE_MX_KB/$F2))
  MEM_SIZE_MAX_KB=$((7*$MEM_SIZE_MX_KB/8))
  export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MAX_KB}k -server"
  echo "MAVEN_OPTS=$MAVEN_OPTS"
}

function get_metric_value {
  fileName="$1"
  metrName="$2"
  fgrep "$metrName" "$fileName" | awk -F: '{print $2}' | sed 's/^\s*//'
}

function getNumCpuCores {
  OS=$(getOS)
  if [ "$OS" = "Linux" ] ; then
    NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
    check "getting the number of CPU cores, do you have /proc/cpu/info?"
  elif [ "$OS" = "Darwin" ] ; then
    NUM_CPU_CORES=4
  else
    echo "Unsupported OS: $OS" 1>&2
    exit 1
  fi
  echo $NUM_CPU_CORES
}

# This function:
# 1. Identifies guesses what is the format of data: new JSONL or old series-of-XML format
# 2. Finds all sub-directories containing indexable data and makes a string 
#    that represents a list of comma-separated sub-directories with data. This string
#    can be passed to indexing apps.
# Attention: it "returns" an array by setting a variable retVal (ugly but works reliably)
function getIndexDataInfo {
  topDir="$1"
  dirList=""
  oldFile="SolrAnswerFile.txt"
  newFile="AnswerFields.jsonl"
  dataFileName=""
  currDir="$PWD"
  cd "$topDir"
  for subDir in * ; do
    if [ -d "$subDir" ] ; then
      if [ -f "$subDir/$oldFile" -o -f "$subDir/$newFile" ] ; then
        if [ "$dirList" != "" ] ; then
          dirList="$dirList,"
        fi
        dirList="${dirList}$subDir"
        if [ -f "$subDir/$oldFile" ] ; then
          if [ "$dataFileName" = "$newFile" ] ; then
            echo "Inconsistent use of XML/JSONL formats"
            exit 1
          fi
          dataFileName=$oldFile
        fi
        if [ -f "$subDir/$newFile" ] ; then
          if [ "$dataFileName" = "$oldFile" ] ; then
            echo "Inconsistent use of XML/JSONL formats"
            exit 1
          fi
          dataFileName=$newFile
        fi
      fi
    fi
  done
  cd "$currDir"
  # This is kinda ugly, but is better than other non-portable solutions.
  retVal=("${dirList}" "${dataFileName}")
}
