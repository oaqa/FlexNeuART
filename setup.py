#!/usr/bin/env python
import os
from os import path
from setuptools import setup, find_packages

PY_CACHE='__pycache__'

curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Solution inspired by https://newbedev.com/how-to-add-package-data-recursively-in-python-setup-py
def get_package_files(root_dir):
    res_list = []
    for (path, dir_list, file_list) in os.walk(root_dir):
        if os.path.split(path)[-1] != PY_CACHE:
            for fn in file_list:
                res_list.append(os.path.join(path, fn))
    return res_list

package_files=get_package_files('flexneuart')

setup(
    name='FlexNeuART',
    description='FlexNeuART (flex-noo-art) is a Flexible classic and NeurAl Retrieval Toolkit',
    python_requires='>=3.6',
    author="Leonid Boytsov",
    author_email="leo@boytsov.info",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0',
    packages=find_packages(exclude=[PY_CACHE]),
    package_data={'': package_files},
    install_requires=[l for l in open('requirements.txt') if not l.startswith('#') and not l.startswith('git+') and l.strip() != ''],
)