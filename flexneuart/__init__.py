import os

import flexneuart

from .utils import *

# Version *MUST* be in Sync with pom.xml
__version__ = '2.0'

def configure_classpath_auto():
    """Automatically configures the class path (see configure_classpath)."""
    root_dir = os.path.dirname(flexneuart.__file__)
    configure_classpath(os.path.join(root_dir, 'resources/jars/'))


def configure_classpath(source_root):
    """Add the FlexNeuART jar to the path. The version of the jar *MUST* match
       the version of the package.

       This function is inspired by Pyserini code https://github.com/castorini/pyserini

    :param source_root: source root
    """

    from jnius_config import set_classpath

    jar_path = os.path.join(source_root, f'FlexNeuART-{__version__}-fatjar.jar')
    if not os.path.exists(jar_path):
        raise Exception(f'JAR file {jar_path} is missing!')

    set_classpath(jar_path)

#
# The Register class is taken from OpenNIR:
# https://github.com/Georgetown-IR-Lab/OpenNIR
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#


class Registry:
    """A decorator that used to register classes."""
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


