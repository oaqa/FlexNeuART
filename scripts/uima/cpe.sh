#!/bin/bash
. scripts/common.sh
setJavaMem 3 4
DescPath=$1
if [ "$DescPath" = "" ] ; then
  echo "Specify a descriptor!"
  exit 1
fi
if [ ! -f "$DescPath" ] ; then
  echo "'$DescPath' is not a file!"
  exit 1
fi
OS=`uname|awk '{print $1}'`
mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.SimpleRunCPE_fixed  -Dexec.args="$DescPath"
if [ "$?" != "0" ] ; then
  echo "Annotation process failed!"
  exit 1
fi
echo "Annotation seem to have finished successfully!"
exit 0
