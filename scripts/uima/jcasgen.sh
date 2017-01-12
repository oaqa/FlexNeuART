#!/bin/bash
mvn exec:java -Dexec.mainClass=org.apache.uima.tools.jcasgen.Jg
if [ "$?" != "0" ] ; then
  exit 1
fi
