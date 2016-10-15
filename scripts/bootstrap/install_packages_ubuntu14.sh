#!/bin/bash

INSTALL_DIR=$1

if [ "$INSTALL_DIR" = "" ] ; then
  echo "Specify an installation directory (1st arg)"
  exit 1
fi

DATA_DIR="$2"

if [ "$DATA_DIR" = "" ] ; then
  echo "Specify a data directory (1st arg)"
  exit 1
fi

BUILD_THREAD_QTY=$3

if [ "$BUILD_THREAD_QTY" = "" ] ; then
  echo "Specify the number of threads to build C/C++ applications, e.g., 4 (3d arg)"
  exit 1
fi

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

function run_cmd {
  cmd="$1"
  bash -c "$cmd"
  check "$cmd"
}

function check_java_check_status {
  f="$?"
  if [ "$f" != "0" ] ; then
    echo "Install Oracle Java 8!"
    exit 1
  fi
}

run_cmd "mkdir -p $INSTALL_DIR"

cd $INSTALL_DIR ; check "cd $INSTALL_DIR"
INSTALL_DIR="$PWD"
cd - ; check "cd - "

run_cmd "mkdir -p $DATA_DIR"

cd $DATA_DIR ; check "cd $DATA_DIR"
DATA_DIR="$PWD"
cd - ; check "cd - "

# This is a very crude check, feel free to
# remove if you are comfortable with using
# a different Java version
which java >/dev/null
check_java_check_status
java -version 2>&1|grep 1.8
check_java_check_status
java 2>&1|grep -i oracle
check_java_check_status

echo "Oracle Java 8 found!"

echo "I am going to install a few packages via 'sudo apt-get install'. You may need to enter a password!"

run_cmd "sudo apt-get update"

MAIN_PACKAGE_LIST="wget g++  \
                      git    \
                      libeigen3-dev libboost1.54-all-dev libgsl0-dev libsparsehash-dev \
                      cmake \
                      mc \
                      maven \
                      wordnet-base \
                      python-scipy \
                      libboost-dev libboost-test-dev \
                      libboost-program-options-dev libboost-system-dev \
                      libboost-filesystem-dev libevent-dev \
                      automake libtool flex bison pkg-config \
                      libssl-dev libboost-thread-dev make"

sudo apt-get install --assume-yes $MAIN_PACKAGE_LIST
check "sudo apt-get install --assume-yes $MAIN_PACKAGE_LIST"

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

run_cmd "git clone https://github.com/moses-smt/giza-pp.git"
echo "GIZA++ is downloaded!"

cd giza-pp ; check "cd giza-pp"

run_cmd "make -j $BUILD_THREAD_QTY"
echo "GIZA++ is compiled!"

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

KNN4QA_BRANCH=""
run_cmd "git clone https://github.com/oaqa/knn4qa.git"
if [ "$KNN4QA_BRANCH" != "" ] ; then
  cd knn4qa ; check "cd knn4qa"
  run_cmd "get fetch"
  run_cmd "get checkout $KNN4QA_BRANCH"
  cd .. ; check "cd .. "
fi
echo "knn4qa is downloaded!"

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

run_cmd "wget http://archive.apache.org/dist/thrift/0.9.2/thrift-0.9.2.tar.gz -O thrift-0.9.2.tar.gz"
echo "Apache Thrift is downloaded"
run_cmd "tar zxvf  thrift-0.9.2.tar.gz"
cd thrift-0.9.2/
check "cd thrift-0.9.2/"
run_cmd "./configure --without-erlang --without-nodejs --without-lua \
                     --without-php --without-ruby --without-haskell \
                     --without-go --without-d"

run_cmd "make -j $BUILD_THREAD_QTY"

echo "Apache Thrift is built. I am going to install it now. You may need to enter a password!"

run_cmd "sudo make install && sudo ldconfig"

echo "Apache Thrift is installed!"

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

run_cmd "git clone https://github.com/searchivarius/nmslib.git"
cd nmslib 
check "cd nmslib"

run_cmd "git fetch"
run_cmd "git checkout nmslib4qa_cikm2016"

cd similarity_search
check "cd similarity_search"
run_cmd "cmake ."
run_cmd "make -j $BUILD_THREAD_QTY"
echo "NMSLIB core is compiled!"
cd ../query_server/cpp_client_server/ ; check "cd ../query_server/cpp_client_server/"
run_cmd "make -j $BUILD_THREAD_QTY"
echo "NMSLIB query server is compiled!"

# Now let's create all directory structure
cd $DATA_DIR ; check "cd $DATA_DIR"

for dir in input output lucene_index memfwdindex WordEmbeddings tran ; do
  run_cmd "mkdir $dir"
  for collect in manner compr ComprMinusManner stackoverflow ; do
    run_cmd "mkdir $dir/$collect"
  done
done
run_cmd "mkdir WordEmbeddings/Complete"

KNN4QA_DIR="$INSTALL_DIR/knn4qa"
NMSLIB_DIR="$INSTALL_DIR/nmslib/similarity_search"

cd "$KNN4QA_DIR" ; check "cd $KNN4QA_DIR"
run_cmd "scripts/data/create_knn4qa_links.sh \"$DATA_DIR\""
run_cmd "scripts/data/create_output_dirs.sh"
run_cmd "mvn compile"
run_cmd "ln -s /usr/share/wordnet"

echo "Links from data directory to $KNN4QA_DIR are created!"

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

run_cmd "git clone https://github.com/searchivarius/clearnlp-clearnlp-2.0.2.mod.git"
cd clearnlp-clearnlp-2.0.2.mod ; check "cd clearnlp-clearnlp-2.0.2.mod"
run_cmd "mvn compile"
run_cmd "mvn install"

echo "Installed a patched version of Clear NLP 2.0.2!"

# Installing trec_eval

# Returning to software home dir
cd $INSTALL_DIR ; check "cd $INSTALL_DIR"

run_cmd "wget https://github.com/usnistgov/trec_eval/archive/v9.0.4.tar.gz -O v9.0.4.tar.gz"
run_cmd "tar -zxvf v9.0.4.tar.gz"
cd trec_eval-9.0.4/ ; check "cd trec_eval-9.0.4/"
run_cmd "make -j $BUILD_THREAD_QTY"
cd "$KNN4QA_DIR" ; check "cd $KNN4QA_DIR"
run_cmd "ln -s $INSTALL_DIR/trec_eval-9.0.4/"

echo "trec_eval is installed!"

# Returning to data home dir
cd "$DATA_DIR" ; check "cd $DATA_DIR"

run_cmd "mkdir nmslib"
cd "nmslib" ; check "cd nmslib"

for collect in compr stackoverflow ; do
  run_cmd "mkdir -p $collect/pivots"
  run_cmd "mkdir -p $collect/queries"
  run_cmd "mkdir -p $collect/index"
  run_cmd "$KNN4QA_DIR/scripts/data/create_nmslib4qa_links.sh \"$DATA_DIR\" \"$INSTALL_DIR\" $collect"
done

cd $NMSLIB_DIR ; check "cd $NMSLIB_DIR"
run_cmd "ln -s $DATA_DIR/nmslib"
cd $KNN4QA_DIR ; check "cd $KNN4QA_DIR"
run_cmd "ln -s $DATA_DIR/nmslib"

echo "Data directories are created, please, explore $DATA_DIR/data, $KNN4QA_DIR, and $NMSLIB_DIR"
echo "Script finished successfully!"



