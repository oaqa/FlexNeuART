import inspect

import scripts.cedr.modeling as modeling
import scripts.cedr.modeling_model1 as modeling_model1

MODEL_PARAM_LIST = ['dropout', 'bert_large']
MODEL_PARAM_PREF = 'model.'

VANILLA_BERT = 'vanilla_bert'
MODEL1_BERT = 'model1_bert'
MODEL1_LEX  = 'model1_lex'
MODEL1_BERT_TERM_INDEP = 'model1_bert_termind'

MODEL_MAP = {
    VANILLA_BERT: modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}

def add_model_init_basic_args(parser):

    parser.add_argument('--model', metavar='model', help='a model to use',
                        choices=MODEL_MAP.keys(), default='vanilla_bert')


    parser.add_argument(f'--{MODEL_PARAM_PREF}dropout', type=float,
                        default=modeling.DEFAULT_BERT_DROPOUT,
                        metavar='optional model droput',
                        help='optional model droput (not for every model)')

    parser.add_argument('--{MODEL_PARAM_PREF}bert_large',
                        action='store_true',
                        help='Using the BERT large mode instead of a base one')


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
    model.set_grad_checkpoint_param(args.grad_checkpoint_param)

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
