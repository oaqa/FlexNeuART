#/bin/bash
source scripts/common.sh
setJavaMem 1 4
bash_cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.BuildInMemFwdIndexApp -Dexec.args='$@' "
bash -c "$bash_cmd"
if [ "$?" != "0" ] ; then
  exit 1
fi
