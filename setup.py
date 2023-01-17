#!/usr/bin/env python
import os
from os import path
from setuptools import setup, find_packages
import unittest
import subprocess
import sys
import platform

PY_CACHE = '__pycache__'
ROOT_DIR_NAME = 'flexneuart'

REC_FILE='requirements.txt'

try:
    print(subprocess.check_output(f'pip install -r {REC_FILE}'.split()).decode())
except:
    print(f'Failed to install requirements from the file {REC_FILE}')
    sys.exit(1)

# Version *MUST* be in Sync with pom.xml
sys.path.append(ROOT_DIR_NAME)
from version import __version__
from config import DEFAULT_ENCODING, MIN_PYTHON_VERSION


print('Building version:', __version__)

curr_python_ver = tuple([int(s) for s in platform.python_version().split('.')])

print('Current python version:', curr_python_ver)
print('Minimum required python version:', MIN_PYTHON_VERSION)

assert tuple(curr_python_ver) >= tuple(MIN_PYTHON_VERSION), \
                                        "Python version is insufficient, we require at least: " + str(MIN_PYTHON_VERSION)

# We need to build java binaries (and pack scripts) before packing everything
# This script also cleans up vestiges of the previous build
BUILD_SCRIPT="./build.sh"

try:
    print(subprocess.check_output([BUILD_SCRIPT]).decode())
except:
    # Make sure build.log is consistent is the log file name used in the build script
    # That is if you change the name in a build script, it should be changed here as well.
    print(f'The build script {BUILD_SCRIPT} failed, please, check out the log: build.log')
    sys.exit(1)


curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir, 'README.md'), encoding=DEFAULT_ENCODING) as f:
    long_description = f.read()

jar_file=f'resources/jars/FlexNeuART-{__version__}-fatjar.jar'

EXCLUDE_DIRS = 'build data dist lemur-code* lib scripts src target testdata trec_eval*'.split()

setup(
    name=ROOT_DIR_NAME,
    version=__version__,
    description='FlexNeuART (flex-noo-art) is a Flexible classic and NeurAl Retrieval Toolkit',
    python_requires='>=3.6',
    author="Leonid Boytsov",
    author_email="leo@boytsov.info",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # We want to distribute source code as well
    # see https://setuptools.readthedocs.io/en/latest/userguide/miscellaneous.html#setting-the-zip-safe-flag
    zip_safe=False,
    scripts=['flexneuart_install_extra.sh', 'install_extra_flexneuart_main.sh'],
    packages=find_packages(exclude=EXCLUDE_DIRS),
    package_data={ROOT_DIR_NAME: [jar_file, 'resources/extra/scripts.tar.gz']},
    install_requires=[l for l in open('requirements.txt') if not l.startswith('#') and not l.startswith('git+') and l.strip() != '']
)

# setup tools fail to find these tests (when one specifies test_suit arg) and it's not clear why
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(start_dir='./tests', pattern='test*.py')
test_runner = unittest.TextTestRunner()
results : unittest.runner.TextTestResult = test_runner.run(test_suite)
assert not results.errors
assert not results.failures
