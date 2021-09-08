#!/bin/bash -e
set -o pipefail
curr_dir=$PWD

log_file="$curr_dir/main.build.log"
echo "========================"
echo " BUILDING main codebase "
echo " log: $log_file"
mvn -U clean package appassembler:assemble &> $log_file || { echo "Build failed!" ; exit 1 ; }

rm -f flexneuart/resources/jars/*.jar
cp target/FlexNeuART*fatjar.jar flexneuart/resources/jars/

echo "======================="
echo "BUILD IS COMPLETE!"
echo "======================="
