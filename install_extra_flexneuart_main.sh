#!/bin/bash -e
set -o pipefail
dstDir="$1"
[ ! -z "$dstDir" ] || { echo "Specify the target directory! (1st arg)" ; exit 1 ; }

withGiza="$2"
[ ! -z "$withGiza" ]  || { echo "Specify the MGIZA installation flag (2d arg)" ; exit 1 ; }

REPO=$(python -c "from flexneuart import get_jars_location ; print(get_jars_location())") \
  || { echo "import error, did you install flexneuart library?" ; exit 1 ; }

currDir=$PWD
cd $REPO
cd ../extra
EXTRA_LOC=$PWD
cd $currDir

if [ -f "$dstDir" ] ; then
  echo "Target exists, but it's not a directory: $dstDir"
  exit 1
fi

# Being a bit paranoid to avoid removal of the wrong directory
[ -z "$dstDir" ] && { echo "Bug: Empty destination directory!" ; exit ; }
rm -rf "$dstDir"
mkdir -p "$dstDir"

cd "$dstDir"
tar zxvf "$EXTRA_LOC"/scripts.tar.gz

./flexneuart_install_extra_bin.sh "$withGiza"