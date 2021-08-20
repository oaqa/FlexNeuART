#
# Based on the code from the ANCE repository
# (MIT license, which is compatible with the current repo's Apache license):
#
# https://github.com/microsoft/ANCE/blob/master/model/models.py
#
# We assume the model file names/directories are the same as in the script download_ance_model.sh
# We also exclude the MS MARCO MaxP model, because it is only marginally better than FirstP.
#
#
import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig
)

REPRESENT_DIM = 768

MODEL_ANCE_FIRSTP = 'ance_firstp'
MODEL_DPR = 'dpr'
NUM_LABELS=2

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models.
    Each model should define its own query and document embedding functions.
    """
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("query embeddings")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("document embedding")

    def forward(self, query_ids, attention_mask_q,
                input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b):

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        return logit_matrix


class RobertaDot_NLL_LN(EmbeddingMixin, RobertaForSequenceClassification):
    """
      ANCE implementation of the Bi-Encoder.

      We inherit from RobertaModel to use from_pretrained
    """

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, REPRESENT_DIM)
        self.norm = nn.LayerNorm(REPRESENT_DIM)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        assert type(outputs) == BaseModelOutputWithPoolingAndCrossAttentions
        full_emb = outputs.last_hidden_state[:, 0, :]
        return self.norm(self.embeddingHead(full_emb))

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)



class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()

    @classmethod
    def init_encoder(model_class, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return model_class.from_pretrained("bert-base-uncased", config=cfg)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        assert type(outputs) == BaseModelOutputWithPoolingAndCrossAttentions
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(EmbeddingMixin, nn.Module):
    """
        Bi-Encoder model from the Facebook DPR paper.
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.question_model = HFBertEncoder.init_encoder()
        self.ctx_model = HFBertEncoder.init_encoder()
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        return self.question_model(input_ids, attention_mask)

    def body_emb(self, input_ids, attention_mask):
        return self.ctx_model(input_ids, attention_mask)
        

class MSMarcoConfig:
    def __init__(self,
                 name,
                 model,
                 use_mean=True,
                 tokenizer_class=RobertaTokenizer,
                 config_class=RobertaConfig):
        self.name = name
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name=MODEL_ANCE_FIRSTP, model=RobertaDot_NLL_LN),
    MSMarcoConfig(name=MODEL_DPR, model=BiEncoder, tokenizer_class=BertTokenizer, config_class=BertConfig)
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}

def create_ance_firstp(model_chkpt_path):
    """Create an ANCE FirstP model and a tokenizer using a path to the checkpoint.

    :param model_chkpt_path: a a checkpoint path
    :return:  a tuple (model, tokenizer)
    """
    config_obj = MSMarcoConfigDict[MODEL_ANCE_FIRSTP]
    config = config_obj.config_class.from_pretrained(
        model_chkpt_path,
        num_labels=NUM_LABELS,
        finetuning_task=None,
        cache_dir=None
    )
    tokenizer = config_obj.tokenizer_class.from_pretrained(
        model_chkpt_path,
        do_lower_case=True,
        cache_dir=None
    )
    model = config_obj.model_class.from_pretrained(
        model_chkpt_path,
        from_tf=False,
        config=config,
        cache_dir=None
    )

    return model, tokenizer


def create_dpr(model_chkpt_path):
    """Create a DPR model and a tokenizer using a path to the checkpoint.

    :param model_chkpt_path: a a checkpoint path
    :return:  a tuple (model, tokenizer)
    """
    config_obj = MSMarcoConfigDict[MODEL_DPR]
    model = config_obj.model_class()
    tokenizer = config_obj.tokenizer_class.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
        cache_dir=None
    )
    check_point = torch.load(model_chkpt_path)
    model.load_state_dict(check_point['model_dict'], strict=False)

    return model, tokenizer
