import argparse

from flexneuart.config import DEFAULT_DEVICE_GPU
from flexneuart.models.vanilla_bert import VANILLA_BERT
from flexneuart import Registry

model_registry = Registry(default='models')
register = model_registry.register


def add_model_init_basic_args(parser, add_device_name):

    model_list = list(model_registry.registered.keys())
    parser.add_argument('--model', metavar='model',
                        help='a model to use: ' + ' '.join(),
                        choices=model_list, default=VANILLA_BERT)

    parser.add_argument('--init_model_weights',
                        metavar='model weights',
                        help='initial model weights will be loaded in non-strict mode',
                        type=argparse.FileType('rb'), default=None)

    parser.add_argument('--init_model',
                        metavar='initial model',
                        help='previously serialized model',
                        type=argparse.FileType('rb'), default=None)

    if add_device_name:
        parser.add_argument('--device_name', metavar='CUDA device name or cpu', default=DEFAULT_DEVICE_GPU,
                            help='The name of the CUDA device to use')



