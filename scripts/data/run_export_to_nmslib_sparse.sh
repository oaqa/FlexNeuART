#/bin/bash
source scripts/common_proc.sh
setJavaMem 6 9
bash_cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.ExportToNMSLIBSparse -Dexec.args='$@' "
bash -c "$bash_cmd"
if [ "$?" != "0" ] ; then
  exit 1
fi
