#/bin/bash -e
source scripts/common_proc.sh
setJavaMem 4 8
bash_cmd="$MVN_RUN_CMD -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.LuceneIndexer -Dexec.args='$@' "
bash -c "$bash_cmd"
exit $?
