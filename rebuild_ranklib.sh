#!/bin/bash
# This script shouldn't be called during a build
# instead we build the jar and add it to github instead
cd lemur-code-r2792-RankLib-trunk
mvn clean package 
targetDir=../lib/umass/RankLib/2.14.fixed
if [ ! -d "$targetDir" ] ; then
  mkdir -p "$targetDir"
fi
cp target/RankLib-2.14.jar $targetDir/RankLib-2.14.fixed.jar
