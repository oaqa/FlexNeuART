import os
import glob

from .utils import *

def configure_classpath(source_root):
    """Add the latest FlexNeuART jar to the path.

       This function is based on Pyserini code https://github.com/castorini/pyserini

    :param source_root: source root
    """

    from jnius_config import set_classpath

    paths = glob.glob(os.path.join(source_root, 'FlexNeuART-*-fatjar.jar'))
    if not paths:
        raise Exception('No matching jar file found in {}'.format(os.path.abspath(source_root)))

    latest = max(paths, key=os.path.getctime)
    set_classpath(latest)
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


