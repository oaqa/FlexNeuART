#/bin/bash
MEM_SIZE_MX_KB=`free|grep Mem|awk '{print $2}'`
MEM_SIZE_MIN_KB=$((3*$MEM_SIZE_MX_KB/4))
export MAVEN_OPTS="-Xms${MEM_SIZE_MIN_KB}k -Xmx${MEM_SIZE_MX_KB}k -server"
bash_cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.LuceneIndexer -Dexec.args='$@' "
bash -c "$bash_cmd"
if [ "$?" != "0" ] ; then
  exit 1
fi

