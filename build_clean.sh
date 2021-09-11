#!/bin/bash -e
echo "==================================="
echo " Cleaning the previous build       "
rm -rf build dist flexneuart.egg-info/ 
# This should be in sync with setup.py and build.sh
rm -rf target/ flexneuart/resources/
echo "==================================="

