#/bin/bash -e
source scripts/common_proc.sh
NO_MAX=0
setJavaMem 6 8 $NO_MAX
bash_cmd="$MVN_RUN_CMD -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.BuildFwdIndexApp -Dexec.args='$@' "
bash -c "$bash_cmd"
exit $?
