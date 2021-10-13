#!/bin/bash -e
echo "==================================="
echo " Cleaning the previous build       "
JAVA_SRC_PREF=java
rm -rf build dist flexneuart.egg-info/
# This should be in sync with setup.py and build.sh
rm -rf ${JAVA_SRC_PREF}/target/ flexneuart/resources/
echo "==================================="

