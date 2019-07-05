#/bin/bash
bash_cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.SQuADCollectionSplitter -Dexec.args='$@' "
bash -c "$bash_cmd"
exit $?

