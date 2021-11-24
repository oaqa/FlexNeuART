#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#
# Helper functions for distributed training
#
import multiprocessing
import os

import torch
import torch.distributed as dist

from typing import Dict, List
from flexneuart.models.train.amp import get_amp_processors

from multiprocessing import Process

# 5 minutes should be more than enough while waiting
# for other processes to reach the same training point
BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT=60*5

MAIN_PROC_FLAG_ARG= 'is_main_proc'

def avg_model_params(model, amp):
    """
       Average model parameters across all GPUs.
       Set amp to True, to enable automatic mixed-precision.
    """
    auto_cast_class, scaler = get_amp_processors(amp)

    with auto_cast_class():
        qty = float(dist.get_world_size())
        for prm in model.parameters():
            dist.all_reduce(prm.data, op=torch.distributed.ReduceOp.SUM)
            prm.data /= qty


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


def get_device_name_arr(device_qty, device_name=None):
    """Populate an array with device names. If the device_name is given,
       we only use this device.

    :param device_qty:      a number of devices
    :param device_name:     an optional name of the device
    :return:
    """
    res = []

    if device_name is not None and device_qty == 1:
        return [device_name]

    for rank in range(device_qty - 1, -1, -1):
        res.append(f'cuda:{rank}')
    return res


def wrapper_func(rank,
                 proc_qty,
                 work_func,
                 param_dict,
                 master_port=None,
                 distr_backend=None):
    """A wrapper function to run a given "work-horse" function in a distributed model.
       When the specified number of processes is <= 1, it initializes PyTorch distributed
       environment and executes the work-horse function. In any case, we create only proc - 1
       processes and the starting (main) process still executes the work-horse function.
       Whenever the work-horse function is excecuted, it gets a flag (is_main_proc),
       which indicates whether the process is the main one or not.

    :param rank:            a process rank
    :param proc_qty:        a total number of processes
    :param work_func:       a work-horse function
    :param param_dict:      a parameter dictionary, which includes
    :param master_port:     optional master port (mandatory if proc_qty > 1)
    :param distr_backend:   optional distributed backend type (mandatory if proc_qty > 1)

    :return: whatever the work-horse function returns
    """

    if proc_qty > 1:
        assert master_port is not None, 'master_port is required in the distributed mode'
        assert distr_backend is not None, 'distr_backend is required in the distributed mode'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(distr_backend, rank=rank, world_size=proc_qty)

    return work_func(**param_dict)


def run_distributed(work_func,
                    shared_params : Dict,
                    proc_specific_params : List[Dict] = None,
                    master_port=None,
                    distr_backend=None,
                    proc_qty=1):
    """A helper function for transparent execution of a "work-horse" function in a single- or multi-processing mode.

    :param work_func:               a work-horse function
    :param shared_params:           parameters that are the same for all processes
    :param proc_specific_params:    parameters that depend on the process rank
    :param master_port:             optional master port (mandatory if proc_qty > 1)
    :param distr_backend:           optional distributed backend type (mandatory if proc_qty > 1)
    :param proc_qty:                an optional number of processes

    :return:  whatever is returned by the work-horse function executed in the main process
    """
    assert len(proc_specific_params) == proc_qty or proc_specific_params is None

    processes = []

    main_proc_ret = None

    # We iterate in reverse so the main process is executed last
    # *AFTER* all other processes are started
    for rank in range(proc_qty - 1, -1, -1):
        param_dict = {}
        for k, v in shared_params.items():
            param_dict[k] = v
        for k, v in proc_specific_params[rank].items():
            assert not k in param_dict, \
                                f'Parameter {k} is present in both shared and process-specific parameter dictionaries!'
            param_dict[k] = v
        assert not MAIN_PROC_FLAG_ARG in param_dict, f'One should not specify {MAIN_PROC_FLAG_ARG} in parameter dictionaries!'
        is_main_proc = (rank == 0)
        param_dict[MAIN_PROC_FLAG_ARG] = is_main_proc
        wrapper_param_dict = {
            'rank': rank,
            'proc_qty': proc_qty,
            'work_func': work_func,
            'param_dict': param_dict,
            'master_port': master_port,
            'distr_backend': distr_backend
        }
        if proc_qty > 1 and not is_main_proc:
            p = Process(target=wrapper_func, kwargs=wrapper_param_dict)
            p.start()
            processes.append(p)
        else:
            main_proc_ret = wrapper_func(**wrapper_param_dict)

    if proc_qty > 1:
        for p in processes:
            join_and_check_stat(p)

        dist.destroy_process_group()

    return main_proc_ret

