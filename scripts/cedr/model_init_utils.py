import inspect
import argparse

import scripts.cedr.data as data
import scripts.cedr.modeling as modeling

MODEL_PARAM_LIST = ['dropout', 'bert_large']
MODEL_PARAM_PREF = 'model.'

VANILLA_BERT = 'vanilla_bert'

MODEL_MAP = {
    VANILLA_BERT: modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}

def add_model_init_basic_args(parser, add_train_params):

    parser.add_argument('--model', metavar='model',
                        help='a model to use: ' + ' '.join(list(MODEL_MAP.keys())),
                        choices=MODEL_MAP.keys(), default='vanilla_bert')

    parser.add_argument('--init_model_weights',
                        metavar='model weights', help='initial model weights',
                        type=argparse.FileType('rb'), default=None)

    parser.add_argument('--init_model',
                        metavar='initial model',
                        help='initial *COMPLETE* model with heads and extra parameters',
                        type=argparse.FileType('rb'), default=None)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')


    parser.add_argument('--device_name', metavar='CUDA device name or cpu', default='cuda:0',
                        help='The name of the CUDA device to use')


    if add_train_params:

        parser.add_argument(f'--{MODEL_PARAM_PREF}dropout', type=float,
                            default=modeling.DEFAULT_BERT_DROPOUT,
                            metavar='optional model droput',
                            help='optional model droput (not for every model)')

        parser.add_argument('--grad_checkpoint_param', type=int, default=0,
                            metavar='grad. checkpoint param',
                            help='gradient checkpointing param (0, no checkpointing, 2 every other layer, 3 every 3rd layer, ...)')

        parser.add_argument(f'--{MODEL_PARAM_PREF}bert_large',
                            action='store_true',
                            help='use the BERT large mode instead of a base one')

        parser.add_argument(f'--{MODEL_PARAM_PREF}vocab_file',
                            metavar='vocabulary file',
                            type=str, default=None,
                            help='a previously built vocabulary file')


        parser.add_argument(f'--{MODEL_PARAM_PREF}prob_network_type',
                            metavar='prob network type',
                            default=modeling_model1.ProbNetworkSumFC.name(),
                            help='a network type to compute probabilities: ' + ' '.join(modeling_model1.PROB_NETWORK_TYPE_NAMES),
                            choices=modeling_model1.PROB_NETWORK_TYPE_NAMES)

        parser.add_argument(f'--{MODEL_PARAM_PREF}use_fasttext',
                            action='store_true',
                            help='use FastText embeddings to initialize lexical Model1 embeddings (dim. defined by FastText)')

        parser.add_argument(f'--{MODEL_PARAM_PREF}no_fasttext_embed_dim',
                            type=int,
                            metavar='embedding dim',
                            default=512,
                            help='Dimensionality of the lexical neural Model1')

        parser.add_argument(f'--{MODEL_PARAM_PREF}prob_self_tran', type=float, default=0.05,
                            metavar='self-train prob',
                            help='self-translation probability of the lexical neural Model1')

        parser.add_argument(f'--{MODEL_PARAM_PREF}proj_dim', type=int, default=128,
                            metavar='model1 projection dim',
                            help='neural lexical model1 projection dimensionionality')


def get_model_param_dict(args, model_class):
    """This function iterates over the list of arguments starting with model.
       and checks if the model constructor supports them. It creates
       a parameter dictionary that can be used to initialize the model.
       Limitation: it supports only regular arguments."""
    param_dict = {}
    arg_vars = vars(args)
    model_init_args = inspect.signature(model_class).parameters.keys()


    for k, v in arg_vars.items():
        if k.startswith(MODEL_PARAM_PREF):
            k = k[len(MODEL_PARAM_PREF):]
            if k in model_init_args:
                arg_vars[k] = v

    return arg_vars


def create_model_from_args(args):
    model_class = MODEL_MAP[args.model]
    model_args = get_model_param_dict(args, model_class)
    model = model_class(**model_args)

    return model


def get_model_param_dict(args, model_class):
    """This function iterates over the list of arguments starting with model.
       and checks if the model constructor supports them. It creates
       a parameter dictionary that can be used to initialize the model.
       Limitation: it supports only regular arguments."""
    param_dict = {}
    arg_vars = vars(args)
    model_init_args = inspect.signature(model_class).parameters.keys()

    for k, v in arg_vars.items():
        if k.startswith(MODEL_PARAM_PREF):
            k = k[len(MODEL_PARAM_PREF):]
            if k in model_init_args:
                param_dict[k] = v

    print('Using the following model parameters for model class %s ' % model_class, param_dict)

    return param_dict
