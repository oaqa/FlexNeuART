#/bin/bash
. scripts/common_proc.sh
setJavaMem 5 9
cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.QueryAppMultThread -Dexec.args='$@' "
execAndCheck "$cmd"
