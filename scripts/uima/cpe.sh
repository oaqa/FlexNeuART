#!/bin/bash
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
if [ "$OS" = "Linux" ] ; then
  MEM_SIZE_MX_KB=`free|grep Mem|awk '{print $2}'`
  MEM_SIZE_MIN_KB=$((4*$MEM_SIZE_MX_KB/5))
  #export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MX_KB}k -server"
  # No mx
  export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -server"
elif [ "$OS" = "Darwin" ] ; then
  # Assuming Macbook pro
  MEM_SIZE_MX_KB=$((16384*1024))
  MEM_SIZE_MIN_KB=$((3*$MEM_SIZE_MX_KB/4))
  #export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MX_KB}k -server"
  export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -server"
fi
mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.SimpleRunCPE_fixed  -Dexec.args="$DescPath"
if [ "$?" != "0" ] ; then
  echo "Annotation process failed!"
  exit 1
fi
echo "Annotation seem to have finished successfully!"
exit 0
