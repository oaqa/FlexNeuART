#/bin/bash
source scripts/common_proc.sh
setJavaMem 5 9
bash_cmd="$MVN_RUN_CMD -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.QueryAppMultThread -Dexec.args='$@' "
bash -c "$bash_cmd"
exit $?
