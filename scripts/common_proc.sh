# Some common/constants function to share,
# This should be consistent with common_proc.sh and config.py

CAND_PROV_LUCENE="lucene"
CAND_PROV_NMSLIB="nmslib"
FWD_INDEX_TYPES="mapdb, lucene, flatdata"
ANSWER_FILE="AnswerFields.jsonl"
QUESTION_FILE="QuestionFields.jsonl"
SAMPLE_COLLECT_ARG="Specify a collection sub-directory, e.g., msmarco_pass"


# Just lowercasing, solution https://stackoverflow.com/a/2264537
function lower {
  arg="$1"
  echo "$arg" | tr '[:upper:]' '[:lower:]'
}

# Replace a series of white-space with a single space
function whiteSpacesToSpace {
  echo $1 | sed -E 's/[[:space:]]+/ /g'
}

# Extract a field from the list of white-space separated fields
function extractFieldValue {
  fn="$2"
  line=`whiteSpacesToSpace "$1"`
  echo "$line" | cut -d " " -f $fn
}

# This command may initiate compilation,
# but due to -o and -q switch it:
# 1) It works offline (i.e, it doesn't print a bunch of annoying messages caused by Maven re-checking dependencies status)
# 3) Works in a quite mode, i.e., doesn't print other warnings and stuff.
#export MVN_RUN_CMD="mvn -q -o compile exec:java "
export MVN_RUN_CMD="mvn -o compile exec:java "


# I think the use of check, checkPipe, and execAndCheck should be limited,
# especially for execAndCheck, as it prevents proper escaping of arguments (with say spaces).
# It is advised to use "set -euxo pipefail" or "set -euxo pipefail" after are done using grep
# for some sort of control flow. Thus, there are cases when execAndCheck is still useful.
#
# Notes:
# 1. set -u aborts execution when uninitialized variables are used
# 2. set -eo pipefail aborts execution when the command fails, including all the piped commands
# 3. set -x prints executed commands
#
# Info: https://coderwall.com/p/fkfaqq/safer-bash-scripts-with-set-euxo-pipefail
function execAndCheck {
  cmd0="$1"
  desc="$2"
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
      echo "  Expanded cmd that was actually run in a separate shell:"
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

function checkPipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
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


function waitChildren {
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

# A hacky procedure to start a CEDR server
# one can specify either initial model weights or the complete initial model to load.
# All other parameters are required.
# IMPORTANT NOTE:
# don't use with:
# set -eo pipefail
function startCedrServer {
  modelType="$1"
  initModelWeights="$2"
  initModel="$3"
  maxQueryLen="$4"
  maxDocLen="$5"
  deviceName="$6"
  port="$7"
  serverPidFile="$8"

  if [ "$initModel" = "" ] ; then
    checkVarNonEmpty "modelType"
    checkVarNonEmpty "initModelWeights"

    initModelArg=" --model $modelType --init_model_weights $initModelWeights"
    initFile="$initModelWeights"
  else
    checkVarNonEmpty "initModel"

    initModelArg=" --init_model  $initModel"
    initFile="$initModel"
  fi

  logFileName=`echo $initFile|sed s'|/|_|g'`
  logFileName="log.$logFileName"

  checkVarNonEmpty "maxQueryLen"
  checkVarNonEmpty "maxDocLen"
  checkVarNonEmpty "deviceName"
  checkVarNonEmpty "port"
  checkVarNonEmpty "serverPidFile"

  # Note -u
  python -u scripts/py_featextr_server/cedr_server.py \
    --max_query_len $maxQueryLen \
    --max_doc_len $maxDocLen \
    --device_name $deviceName \
    --port $port \
    $initModelArg \
    &> $logFileName    &

  PID=$!

  if [ "$?" != "0" ] ; then
    echo "Starting CEDR immediate failure, check out $logFileName"
    exit 1
  fi

  echo $PID > "$serverPidFile"


  started=0
  while [ "$started" = "0" ]
  do
    # Usuall the server starts quite quickly
    sleep 5

    echo "Checking if CEDR server (PID=$PID) has started"
    ps -p $PID &>/dev/null

    if [ "$?" != "0" ] ; then
      echo "CEDR server stopped unexpectedly, check logs: $logFileName"
      exit 1
    fi

    grep -iE "start.*server" $logFileName &>/dev/null

    if [ "$?" = "0" ] ; then
      echo "CEDR server has started!"
      started=1
    fi
  done
}

function getOS {
  uname|awk '{print $1}'
}

function setJavaMem {
  F1="$1"
  F2="$2"
  NO_MAX="$3"
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
  MEM_SIZE_MIN_KB=$(($F1*$MEM_SIZE_MX_KB/$F2))
  MEM_SIZE_MAX_KB=$((7*$MEM_SIZE_MX_KB/8))
  if [ "$NO_MAX" = "1" ] ; then
    export JAVA_OPTS="-Xms${MEM_SIZE_MIN_KB}k -server"
  else
    export JAVA_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MAX_KB}k -server"
  fi
  echo "JAVA_OPTS=$JAVA_OPTS"
}

function getNumCpuCores {
  python -c 'import multiprocessing; print(multiprocessing.cpu_count())'
}

# This function:
# 1. Identifies guesses what is the format of data: new JSONL or old series-of-XML format
# 2. Finds all sub-directories containing indexable data and makes a string 
#    that represents a list of comma-separated sub-directories with data. This string
#    can be passed to indexing (and querying) apps.
# Attention: it "returns" an array by setting a variable retVal (ugly but works reliably)
function getIndexQueryDataInfo {
  topDir="$1"
  indexDirs=""
  oldFile="SolrAnswerFile.txt"
  oldQueryFile="SolrQuestionFile.txt"
  newFile=$ANSWER_FILE
  newQueryFile=$QUESTION_FILE
  dataFileName=""
  queryFileName=""
  currDir="$PWD"
  cd "$topDir"
  for subDir in * ; do
    echo "Checking data sub-directory: $subDir"
    if [ -d "$subDir" ] ; then
      hasData=0
      if [ -f "$subDir/$oldFile" ] ; then
        if [ -f "$subDir/$oldFile" ] ; then
          if [ "$dataFileName" = "$newFile" ] ; then
            echo "Inconsistent use of XML/JSONL formats"
            exit 1
          fi
          dataFileName=$oldFile
          qyeryFileName=$oldQueryFile
          hasData=1
        fi
      fi

      # New-layout/format data may be compressed, but queries shouldn't be compressed (and there's little sense to do so)
      for suff in "" ".gz" ".bz2" ; do
        fn=$subDir/${newFile}${suff}
        if [ -f "$fn" ] ; then
          echo "Found indexable data file: $fn"
          if [ "$dataFileName" = "$oldFile" ] ; then
            echo "Inconsistent use of XML/JSONL formats"
            exit 1
          fi
          dataFileName=${newFile}${suff}
          queryFileName=$newQueryFile
          hasData=1
          break
        fi
      done

      if [ "$hasData" = "1" ] ; then
        if [ "$indexDirs" != "" ] ; then
          indexDirs="$indexDirs,"
        fi
        indexDirs="${indexDirs}$subDir"
      fi
    else
      echo "No $subdir"
    fi # if [ -d "$subDir"]
  done
  queryDirs=""
  for subDir in * ; do
    if [ -d "$subDir" ] ; then
      fn=$subDir/${queryFileName}
      if [ -f "$fn" ] ; then
        echo "Found query file: $fn"
        if [ "queryDirs" != "" ] ; then
          queryDirs="$queryDirs,"
        fi
        queryDirs="${queryDirs}$subDir"
      fi
    fi
  done
  cd "$currDir"
  # This is kinda ugly, but is better than other non-portable solutions.
  retVal=("${indexDirs}" "${dataFileName}" "${queryDirs}" "${queryFileName}")
}

function getCatCmd {
  fileName=$1
  catCommand=""
  if [ -f "$fileName" ] ; then

    # Not all parts correspond to the data files
    echo "$fileName" | grep '^.gz$' >/dev/null
    if [ "$?" = "0" ] ; then
      catCommand="zcat"
    else
      echo "$fileName" | grep '^.bz2$' >/dev/null
      if [ "$?" = "0" ] ; then
        catCommand="zcat"
      else
        catCommand="cat"
      fi
    fi
  fi
  echo $catCommand
}

function removeComment {
  line="$1"

  bash -c "echo \"$line\" | grep -E '^\s*[#]' >/dev/null"

  if [ "$?" = "0" ] ; then
    line=""
  fi


  echo $line
}


#
#
# The genUsage and parseArguments
# make assumptions about how parameter-describing
# variables are stored and presented.
# For simplicity, it expects the following
# arrays to exist:
#
#boolOpts=(\
#"opt1" "variable name 1" "help msg" \
#"opt2" "variable name 2" "help msg" \
#)
#
#paramOpts=(\
#"opt3" "variable name 3" "help msg" \
#"opt4" "variable name 4" "help msg" \
#)
#

function genUsage {
  posArgsUsage=$1
  errorMsg=$2

  boolOptsQty=${#boolOpts[*]}
  paramOptsQty=${#paramOpts[*]}

  if [ "$errorMsg" != "" ] ; then
    echo "$errorMsg"
  fi

  if [ "$boolOptsQty" != "0" -o "$paramOptsQty" != "0" ] ; then

    echo "Usage: $posArgsUsage [additional options]"
    echo "Additional options:"


    for ((i=0;i<$boolOptsQty;i+=3)) ; do
      echo "-${boolOpts[$i]} ${boolOpts[$i+2]}"
    done

    for ((i=0;i<$paramOptsQty;i+=3)) ; do
      echo "-${paramOpts[$i]} ${paramOpts[$i+2]}"
    done
  fi
}

function parseArguments {
  posArgs=()

  boolOptsQty=${#boolOpts[*]}
  paramOptsQty=${#paramOpts[*]}

  for ((i=1;i<$boolOptsQty;i+=3)) ; do
    eval "${boolOpts[$i]}=0"
  done

  while [ $# -ne 0 ] ; do
    if [[ "$1" = -* ]] ; then
      i=0
      f=0
      for ((i=0;i<$boolOptsQty;i+=3)) ; do
        if [ "-${boolOpts[$i]}" = "$1" ] ; then
          eval "${boolOpts[$i+1]}=1"
          shift 1
          f=1
          break
        fi
      done
      if [ "$f" = "1" ] ; then
        continue
      fi
      for ((i=0;i<$paramOptsQty;i+=3)) ; do
        if [ "-${paramOpts[$i]}" = "$1" ] ; then
          eval "${paramOpts[$i+1]}=\"$2\""
          shift 2
          f=1
          break
        fi
      done

      if [ "$f" = "1" ] ; then
        continue
      fi
      echo "Invalid option: $1"
      exit 1
    fi
    posArgs=(${posArgs[*]} "$1")
    shift 1
  done
}

# Hacky function to obtain 1, if the first # is greater than another
function isGreater {
  val1=$1
  val2=$2
  python -c "print(int($val1 > $val2))"
}
