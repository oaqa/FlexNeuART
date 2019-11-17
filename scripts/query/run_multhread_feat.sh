#/bin/bash -e
source scripts/common_proc.sh

setJavaMem 5 9
bash_cmd="$MVN_RUN_CMD -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.GenFeaturesAppMultThread -Dexec.args='$@' "
bash -c "$bash_cmd"
if [ "$?" != "0" ] ; then
  exit 1
fi
