#!/bin/bash -e
set -o pipefail

function usage {
  msg=$1
  echo $msg
  cat <<EOF
Usage: <location to install scripts> [additional options]
Additional options:
  -h          print help
  -with_giza  install MGIZA
EOF
}

withGiza="0"

while [ $# -ne 0 ] ; do
  optValue=""
  opt=""
  if [[ "$1" = -* ]] ; then
    optName="$1"
    if [ "$optName" = "-h" -o "$optName" = "-help" ] ; then
      usage
      exit 1
    elif [ "$optName" = "-with_giza" ] ; then
      withGiza="1"
      shift 1
    else
      echo "Invalid option: $optName" >&2
      exit 1
    fi
  else
    posArgs=(${posArgs[*]} $1)
    shift 1
  fi
done

dstDir=${posArgs[0]}
if [ "$dstDir" = "" ] ; then
  usage "Specify desination directory (1st arg)"
  exit 1
fi

if [ -f "$dstDir" ] ; then
  echo "File exists, but it's not a directory: $dstDir"
  exit 1
fi

if [ ! -d "$dstDir" ] ; then
  mkdir "$dstDir"
fi

cd $dstDir
curr_dir=$PWD
log_file="$curr_dir/install.log"
echo "=========================================="
echo " Installing scripts & additional packages "
echo " Destination directory: $dstDir           "
echo " log: $log_file"

# Must be installed an in the path!
flexneuart_install_extra_main.sh "$withGiza" &> $log_file || { echo "Install failed!" ; exit 1 ; }

echo " INSTALL IS COMPLETE!  "
echo "=========================================="
