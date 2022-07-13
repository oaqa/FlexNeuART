"""
    Code taken from ColBERT (new_api branch), which is distributed under Apache-compatible MIT license.
    Some changes:
    1. tokenizer is now being initialized by model name (bert_flavor) rather than from a checkpoint.
    2. context support is removed
    3. maximum query length is provided via parameter, i.e., config is ignored.

    https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""
from flexneuart.models.colbert.hf_colbert import HF_ColBERT
from flexneuart.models.colbert.config.config import ColBERTConfig
from .utils import _split_into_batches

class QueryTokenizer():
    def __init__(self, bert_flavor: str, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(bert_flavor)

        self.config = config

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, max_query_len, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (max_query_len - (len(lst)+3)) for lst in tokens]

        return tokens

    def tensorize(self, batch_text, max_query_len, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=max_query_len)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
