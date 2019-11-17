#/bin/bash
source scripts/common_proc.sh
bash_cmd="$MVN_RUN_CMD -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.CheckDenseSparseExportScores -Dexec.args='$@' "
bash -c "$bash_cmd"
if [ "$?" != "0" ] ; then
  exit 1
fi
