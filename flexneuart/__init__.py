#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os

from typing import Dict, Callable, List, Union

import flexneuart

from flexneuart.utils import *
from .version import __version__


def get_jars_location():
    """
        Return the location of the JAR files for installed library.
    """
    root_dir = os.path.dirname(flexneuart.__file__)
    return os.path.join(root_dir, 'resources/jars/')


def configure_classpath(jar_dir=None):
    """
        Add the FlexNeuART jar to the path. The version of the jar *MUST* match
       the version of the package. If the directory is explicitly specified, we
       use jar from that directory. Otherwise, we determine it automatically
       (via location of installed packages).

        :param jar_dir: a directory with the JAR file
    """

    from jnius_config import set_classpath

    if jar_dir is None:
        jar_dir = get_jars_location()

    jar_path = os.path.join(jar_dir, f'FlexNeuART-{__version__}-fatjar.jar')
    if not os.path.exists(jar_path):
        raise Exception(f'JAR file {jar_path} is missing!')

    set_classpath(jar_path)


class Registry:
    """
        A decorator that used to register classes.

        It is a slightly modified version of the Register class from OpenNIR:
        https://github.com/Georgetown-IR-Lab/OpenNIR

        (c) Georgetown IR lab & Carnegie Mellon University

        It's distributed under the MIT License
        MIT License is compatible with Apache 2 license for the code in this repo.

    """
    def __init__(self):
        self.registered : Dict[str, Callable] = {}

    def register(self, name_or_names : Union[str, List[str]]):
        registry = self

        if type(name_or_names) == str:
            name_arr = [name_or_names]
        else:
            assert type(name_or_names) == list
            name_arr = name_or_names

        def wrapped(fn):
            for name in name_arr:
                registry.registered[name] = fn

            fn.name = name_arr[0]
            return fn

        return wrapped

