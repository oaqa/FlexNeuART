"""
    Code taken from ColBERT (new_api branch), which is distributed under Apache-compatible MIT license.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
from dataclasses import dataclass

from .base_config import BaseConfig
from .settings import *


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(RunSettings, ResourceSettings, DocSettings, QuerySettings, TrainingSettings,
                    IndexingSettings, SearchSettings, BaseConfig):
    pass
