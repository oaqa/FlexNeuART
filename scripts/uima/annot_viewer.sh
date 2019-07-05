#!/bin/bash
mvn exec:java -Dexec.mainClass=org.apache.uima.tools.AnnotationViewerMain
if [ "$?" != "0" ] ; then
  exit 1
fi
