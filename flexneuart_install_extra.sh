#!/bin/bash -e
set -o pipefail
dstDir="$1"
[ ! -z "$dstDir" ] || { echo "Specify the target directory! (1st arg)" ; exit 1 ; }

withGiza="$2"
[ ! -z "$withGiza" ]  || { echo "Specify the MGIZA installation flag (2d arg)" ; exit 1 ; }

set -o pipefail
curr_dir=$PWD

log_file="$curr_dir/install.log"
echo "=============================================="
echo " Installing additional scripts & binaries "
echo " log: $log_file"

install_extra_flexneuart_main.sh "$dstDir" "$withGiza" &> $log_file || { echo "Install failed!" ; exit 1; }

echo "=============================================="
echo "            INSTALL IS COMPLETE!"
echo "=============================================="
