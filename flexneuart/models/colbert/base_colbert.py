"""
    Code taken from ColBERT (new_api branch), which is distributed under Apache-compatible MIT license.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
import torch

from transformers import AutoTokenizer
from .hf_colbert import HF_ColBERT
from .config import ColBERTConfig


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name, colbert_config=None):
        super().__init__()

        self.name = name
        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name), colbert_config)
        self.model = HF_ColBERT.from_pretrained(name, colbert_config=self.colbert_config)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.bert

    @property
    def linear(self):
        return self.model.linear
    
    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)

        self.colbert_config.save_for_checkpoint(path)

