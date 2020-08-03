#!/bin/bash -e
curr_dir=$PWD
log_file="$curr_dir/trec_eval.build.log"
echo "========================"
echo " BUILDING trec_eval "
echo " log: $log_file"
cd trec_eval  >/dev/null
make  &> $log_file || { echo "Build failed!" ; exit 1 ; }
echo " Success!"
cd - >/dev/null
# This needs to be built before the main code base, which depends on RankLib!
log_file="$curr_dir/RankLib.build.log"
echo "========================"
echo " BUILDING RankLib "
echo " log: $log_file"
cd lemur-code-r2792-RankLib-trunk >/dev/null
mvn clean package &> $log_file || { echo "Build failed!" ; exit 1 ; }
cp target/RankLib-2.14.jar ../lib/umass/RankLib/2.14.fixed/RankLib-2.14.fixed.jar
echo " Success!"
cd - >/dev/null
cd - >/dev/null

log_file="$curr_dir/main.build.log"
echo "========================"
echo " BUILDING main codebase "
echo " log: $log_file"
echo " Success!"
cd - >/dev/null
mvn clean package appassembler:assemble &> $log_file || { echo "Build failed!" ; exit 1 ; }

echo "======================="
echo "BUILD IS COMPLETE!"
echo "======================="
