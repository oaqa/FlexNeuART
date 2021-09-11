import os

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
    def __init__(self, default: str = None):
        self.registered = {}
        self.default = default

    def register(self, name):
        registry = self

        def wrapped(fn):
            registry.registered[name] = fn
            fn.name = name
            return fn
        return wrapped

