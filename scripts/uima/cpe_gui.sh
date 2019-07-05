#!/bin/bash
mvn compile exec:java -Dexec.mainClass=org.apache.uima.tools.cpm.CpmFrame
if [ "$?" != "0" ] ; then
  exit 1
fi
