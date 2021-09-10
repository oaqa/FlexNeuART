#!/bin/bash -e
#
# This scripts builds java binaries.
#
set -o pipefail
curr_dir=$PWD

log_file="$curr_dir/build.log"
echo "========================"
echo " BUILDING main codebase "
echo " log: $log_file"

./build_clean.sh &> $log_file || { echo "Build failed!" ; exit 1; }

./build_main.sh &> $log_file || { echo "Build failed!" ; exit 1; }

echo "======================="
echo "BUILD IS COMPLETE!"
echo "======================="

