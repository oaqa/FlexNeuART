"""
    Code taken from ColBERT (new_api branch), which is distributed under Apache-compatible MIT license.
    Most of the original code is removed, b/c we only need to support loading from an existing checkpoint.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
import os
import ujson
from dataclasses import dataclass
from .core_config import *

@dataclass
class BaseConfig(CoreConfig):
    @classmethod
    def from_existing(cls, *sources):
        kw_args = {}

        for source in sources:
            if source is None:
                continue
                
            local_kw_args = dataclasses.asdict(source)
            local_kw_args = {k: local_kw_args[k] for k in source.assigned}
            kw_args = {**kw_args, **local_kw_args}

        obj = cls(**kw_args)

        return obj

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        loaded_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        if os.path.exists(loaded_config_path):
            loaded_config, _ = cls.from_path(loaded_config_path)
            loaded_config.set('checkpoint', checkpoint_path)

            return loaded_config

        return None  # can happen if checkpoint_path is something like 'bert-base-uncased'

    @classmethod
    def from_deprecated_args(cls, args):
        obj = cls()
        ignored = obj.configure(ignore_unrecognized=True, **args)

        return obj, ignored

    @classmethod
    def from_path(cls, name):
        with open(name) as f:
            args = ujson.load(f)

            if 'config' in args:
                args = args['config']

        return cls.from_deprecated_args(args)  # the new, non-deprecated version functions the same at this level.
