#/bin/bash -e
source scripts/common_proc.sh
setJavaMem 4 8
cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.BuildFwdIndexApp -Dexec.args='$@' "
execAndCheck "$cmd"
