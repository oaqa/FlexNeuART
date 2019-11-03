#/bin/bash -e
source scripts/common_proc.sh
NO_MAX=0
setJavaMem 6 8 $NO_MAX
cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.BuildFwdIndexApp -Dexec.args='$@' "
execAndCheck "$cmd"
