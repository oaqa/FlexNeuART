import torch
import random
import numpy
import multiprocessing

PYTORCH_DISTR_BACKEND='gloo'

def set_all_seeds(seed):
    """Just set the seed value for common packages including the standard random."""
    print(f'Setting the seed to {seed}')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def enable_spawn():
    """Enable light-weight children. Plus, it is
       a must-use process createion mode for multi-GPU training.
    """
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def join_and_check_stat(proc):
    """Join the process and check its status:
       Raise an exception when a sub-process exits abnormally (exit status != 0).
    """
    proc.join()
    if proc.exitcode != 0:
        raise Exception('A process exited abnormally with code:' + str(proc.exitcode))
