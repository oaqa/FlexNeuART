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

function usage {
  err=$1
  echo "$err"
  echo "Usage <giza/mgiza dir> <work dir> <source file> <target file> <num iter>"
  exit 1
}

giza_dir="$1"

if [ "$giza_dir" = "" ] ; then
  usage "Specify <giza/mgiza dir>"
fi
if [ ! -d "$giza_dir" ] ; then
  usage "Not a directory specified: '$giza_dir'"
fi

work_dir="$2"

if [ "$work_dir" = "" ] ; then
  usage "Specify <work dir>"
fi
if [ ! -d "$work_dir" ] ; then
  usage "Not a directory specified: '$work_dir'"
fi

source_file="$3"

if [ "$source_file" = "" ] ; then
  usage "Specify <source file>"
fi

if [ ! -f "$source_file" ] ; then
  usage "Not a file $source_file"
fi

target_file="$4"

if [ "$target_file" = "" ] ; then
  usage "Specify <target file>"
fi

if [ ! -f "$target_file" ] ; then
  usage "Not a file $target_file"
fi

num_iter=$5

if [ "$num_iter" = "" ] ; then
  usage "Specify <num iter>"
fi

cd "$work_dir"
check "cd '$work_dir'"

rm -f source
rm -f target

echo "Soft-linking source/target files"


ln -s "$source_file" "source"
check "ln -s $source_file source"
ln -s "$target_file" "target"
check "ln -s $target_file target"

echo "Linking done"

