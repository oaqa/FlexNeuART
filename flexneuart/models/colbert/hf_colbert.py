"""
    Code taken from ColBERT (new_api branch), which is distributed under Apache-compatible MIT license.
    This version is slightly modified: torch_load_dnn is removed.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

class HF_ColBERT(BertPreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.dim = colbert_config.dim
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
        obj.base = name_or_path

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj


