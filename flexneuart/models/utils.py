#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import math
import torch
import argparse

from transformers import PreTrainedTokenizerBase

from flexneuart.config import DEFAULT_DEVICE_GPU
from transformers import AutoTokenizer, AutoModel
from flexneuart.models import model_registry

# An attribute to store the main BERT encoder
BERT_ATTR='bert'

def is_longformer(bert_flavor: str):
    """
    A very hacky check if the model is a longformer-type model.

    :param bert_flavor:   the name of the underlying BERT Transformer
    :return:
    """
    return bert_flavor.lower().find('longformer') >= 0

def init_model(obj_ref, bert_flavor : str):
    """Instantiate a model, a tokenizer, and remember their parameters.

    :param obj_ref:       an object to initialize.
    :param bert_flavor:   the name of the underlying BERT Transformer
    """

    obj_ref.BERT_MODEL = bert_flavor

    model = AutoModel.from_pretrained(bert_flavor)

    config = model.config
    setattr(obj_ref, BERT_ATTR, model)
    obj_ref.config = config

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(bert_flavor)
    obj_ref.tokenizer = tokenizer
    obj_ref.no_token_type_ids = not 'token_type_ids' in tokenizer.model_input_names

    obj_ref.CHANNELS = config.num_hidden_layers + 1
    obj_ref.BERT_SIZE = config.hidden_size
    obj_ref.MAXLEN = config.max_position_embeddings

    obj_ref.CLS_TOK_ID = tokenizer.cls_token_id
    obj_ref.SEP_TOK_ID = tokenizer.sep_token_id

    print('Model type:', obj_ref.BERT_MODEL,
          '# of channels:', obj_ref.CHANNELS,
          'hidden layer size:', obj_ref.BERT_SIZE,
          'input window size:', obj_ref.MAXLEN,
          'no token type IDs:', obj_ref.no_token_type_ids)


#
# This function should produce averaging coefficients compatible with the split produced by subbatch
#
def get_batch_avg_coeff(mask, maxlen):
    # Fortunately for us, the mask type is float or else division would
    # have resulted in zeros as 1.0 / torch.LongTensor([4]) == 0
    return 1.0/torch.ceil(torch.sum(mask, dim=-1) / maxlen)


def subbatch(toks, maxlen):
    """Splits tokens into sub-batches each containing at most maxlen tokens."""
    assert(maxlen > 0)
    _, dlen = toks.shape[:2]
    subbatch_qty = math.ceil(dlen / maxlen)
    S = math.ceil(dlen / subbatch_qty) if subbatch_qty > 0 else 0 # minimize the size given the number of subbatches
    stack = []
    if subbatch_qty == 1:
        return toks, subbatch_qty
    else:
        for s in range(subbatch_qty):
            stack.append(toks[:, s*S:(s+1)*S])
            if stack[-1].shape[1] != S:
                nulls = torch.zeros_like(toks[:, :S - stack[-1].shape[1]])
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        return torch.cat(stack, dim=0), subbatch_qty


def un_subbatch(embed, toks, subbatch_qty):
    """Reverts the subbatching"""
    batch, dlen = toks.shape[:2]

    if subbatch_qty == 1:
        return embed
    else:
        embed_stack = []
        for b in range(subbatch_qty):
            embed_stack.append(embed[b*batch:(b+1)*batch])
        embed = torch.cat(embed_stack, dim=1)
        embed = embed[:, :dlen]
        return embed


def sliding_window_subbatch(toks, window_size, stride):
    """A sliding-window sub-batching function.

    :param toks:            batched (and encoded) input tokens
    :param window_size:     a sliding window size
    :param stride:          a sliding window stride

    :return:  a tuple (stack sub-batched tokens, the number of sub-batches)
    """
    _, dlen = toks.shape[:2]
    assert dlen > 0
    # Ceiling of the negative number is a negative number too!!!
    # Hence, max(0, ...
    subbatch_qty = math.ceil(max(0, dlen-window_size)/stride) + 1
    assert subbatch_qty > 0, f'Bad sub-batch: {subbatch_qty} dlen {dlen} window_size {window_size} stride {stride}'
    stack = []
    if subbatch_qty == 1:
       return toks, subbatch_qty
    else:
        for s in range(subbatch_qty):
            if s*stride+window_size < dlen:
                stack.append(toks[:, s*stride: s*stride+window_size])
            else:
                nulls = torch.zeros_like(toks[:, :s*stride+window_size - dlen])
                stack.append(torch.cat([toks[:, s*stride:], nulls], dim=1))

        return torch.cat(stack, dim=0), subbatch_qty


def add_model_init_basic_args(parser, add_device_name, add_init_model_weights, mult_model):
    parser.add_argument('--amp', action='store_true',
                        help="Use automatic mixed-precision")

    model_list = list(model_registry.registered.keys())

    if add_init_model_weights:
        parser.add_argument('--init_model_weights',
                            metavar='model weights',
                            help='initial model weights will be loaded in non-strict mode',
                            type=str, default=None)

    if not mult_model:
        parser.add_argument('--model_name', metavar='model_name',
                            help='a model to use: ' + ', '.join(model_list),
                            choices=model_list,
                            default=None)

        parser.add_argument('--init_model',
                            metavar='initial model',
                            help='previously serialized model',
                            type=str, default=None)
    else:
        parser.add_argument('--init_model_list',
                            metavar='serialized models',
                            required=True,
                            help='previously serialized models',
                            type=str,
                            nargs='+',
                            default=None)

    if add_device_name:
        parser.add_argument('--device_name', metavar='CUDA device name or cpu', default=DEFAULT_DEVICE_GPU,
                            help='The name of the CUDA device to use')
