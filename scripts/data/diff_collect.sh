#/bin/bash
bash_cmd="mvn compile exec:java -Dexec.mainClass=edu.cmu.lti.oaqa.knn4qa.apps.CollectionDiffer -Dexec.args='$@' "
bash -c "$bash_cmd"

