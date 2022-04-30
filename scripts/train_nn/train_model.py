#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR: https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
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
import os
import time
import sys
import math
import argparse
import numpy as np

from transformers.optimization import get_constant_schedule_with_warmup
from threading import BrokenBarrierError
from multiprocessing import Barrier
from typing import List

import flexneuart.config
import flexneuart.io.train_data

from flexneuart.models.utils import add_model_init_basic_args

from flexneuart.models.train import run_model, clean_memory
from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.models.train.batch_obj import BatchObject
from flexneuart.models.train.batching import TrainSamplerFixedChunkSize,\
                                             BatchingTrainFixedChunkSize

from flexneuart.models.train.distr_utils import run_distributed, get_device_name_arr, \
                                                enable_spawn, avg_model_params, \
                                                BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT
from flexneuart.models.train.loss import *
from flexneuart.models.train.amp import get_amp_processors

from flexneuart.data_augmentation.augmentation_module import *

from flexneuart import sync_out_streams, set_all_seeds
from flexneuart.io.json import read_json, save_json
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.eval import METRIC_LIST, get_eval_results

from flexneuart.config import TQDM_FILE

from tqdm import tqdm
from collections import namedtuple

OPT_SGD = 'sgd'
OPT_ADAMW = 'adamw'

VALID_ALWAYS = 'always'
VALID_LAST = 'last_epoch'
VALID_NONE = 'never'

LR_SCHEDULE_CONST = 'const'
LR_SCHEDULE_CONST_WARMUP = 'const_with_warmup'
LR_SCHEDULE_ONE_CYCLE_LR = 'one_cycle'

TrainParams = namedtuple('TrainParams',
                    ['optim',
                     'device_name',
                     'init_lr', 'init_bert_lr', 'epoch_lr_decay', 'weight_decay',
                     'momentum',
                     'amp',
                     'warmup_pct', 'lr_schedule',
                     'batch_sync_qty',
                     'batches_per_train_epoch',
                     'batch_size', 'batch_size_val',
                     'max_query_len', 'max_doc_len',
                     'cand_score_weight', 'neg_qty_per_query',
                     'backprop_batch_size',
                     'epoch_qty', 'epoch_repeat_qty',
                     'save_epoch_snapshots',
                     'print_grads',
                     'shuffle_train',
                     'valid_type',
                     'use_external_eval', 'eval_metric'])


def get_lr_desc(optimizer):
    lr_arr = ['LRs:']
    for param_group in optimizer.param_groups:
        lr_arr.append('%.6f' % param_group['lr'])

    return ' '.join(lr_arr)


def train_iteration(model_holder, device_name,
                    sync_barrier, sync_qty_target,
                    lr, bert_lr,
                    is_main_proc, device_qty,
                    loss_obj,
                    train_params,
                    dataset, train_pairs, qrels):

    bert_param_keys = model_holder.model.bert_param_names()
    model = model_holder.model

    all_params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    # BERT parameters use a special learning weight
    bert_params = {'params': [v for k, v in all_params if k in bert_param_keys], 'lr': bert_lr}
    non_bert_params = {'params': [v for k, v in all_params if not k in bert_param_keys]}

    if train_params.optim == OPT_ADAMW:
        optimizer = torch.optim.AdamW([non_bert_params, bert_params],
                                      lr=lr, weight_decay=train_params.weight_decay)
    elif train_params.optim == OPT_SGD:
        optimizer = torch.optim.SGD([non_bert_params, bert_params],
                                    lr=lr, weight_decay=train_params.weight_decay,
                                    momentum=train_params.momentum)
    else:
        raise Exception('Unsupported optimizer: ' + train_params.optim)

    max_train_qty = flexneuart.io.train_data.train_item_qty_upper_bound(train_pairs, train_params.epoch_repeat_qty)
    lr_steps = int(math.ceil(max_train_qty / train_params.batch_size))
    scheduler = None
    lr_schedule= train_params.lr_schedule
    if lr_schedule == LR_SCHEDULE_CONST:
        if train_params.warmup_pct:
            raise Exception('Warm-up cannot be used with LR schedule: ' + lr_schedule)
    elif lr_schedule in [LR_SCHEDULE_CONST_WARMUP, LR_SCHEDULE_ONE_CYCLE_LR]:
        if not train_params.warmup_pct:
            raise Exception('LR schedule: ' + lr_schedule + ' requires a warm-up parameter!')

        num_warmup_steps = int(train_params.warmup_pct * lr_steps)

        if lr_schedule == LR_SCHEDULE_ONE_CYCLE_LR:
            if is_main_proc:
                print(f'Using a one-cycle scheduler with a warm-up for {num_warmup_steps} steps')
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            total_steps=lr_steps,
                                                            max_lr=[lr, bert_lr],
                                                            anneal_strategy='linear',
                                                            pct_start=train_params.warmup_pct)
        else:
            assert lr_schedule == LR_SCHEDULE_CONST_WARMUP
            if is_main_proc:
                print(f'Using a const-learning rate scheduler with a warm-up for {num_warmup_steps} steps')

            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    else:
        raise Exception('Unsupported LR schedule: ' + lr_schedule)

    if is_main_proc:
        tqdm.write('Optimizer:' + str( optimizer))

    clean_memory(device_name)
    model.to(device_name)

    model.train()
    total_loss = 0.
    total_prev_qty = total_qty = 0. # This is a total number of records processed, it can be different from
                                    # the total number of training pairs

    batch_size = train_params.batch_size

    optimizer.zero_grad()

    lr_desc = get_lr_desc(optimizer)

    batch_id = 0

    if is_main_proc:

        sync_out_streams()

        if train_params.print_grads:
            tqdm.write('Gradient sums before training')
            for k, v in model.named_parameters():
                tqdm.write(k + ' ' + str('None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2))))

        pbar = tqdm('training', total=max_train_qty, ncols=80, desc=None, leave=False, file=TQDM_FILE)
    else:
        pbar = None

    if loss_obj.has_mult_negatives():
        neg_qty_per_query = train_params.neg_qty_per_query
        assert neg_qty_per_query >= 1
    else:
        neg_qty_per_query = 1

    cand_score_weight = torch.FloatTensor([train_params.cand_score_weight]).to(device_name)

    auto_cast_class, scaler = get_amp_processors(train_params.amp)

    train_sampler = TrainSamplerFixedChunkSize(train_pairs=train_pairs,
                                               neg_qty_per_query=neg_qty_per_query,
                                               qrels=qrels,
                                               epoch_repeat_qty=train_params.epoch_repeat_qty,
                                               do_shuffle=train_params.shuffle_train)

    data_augment_method = None
    if train_params.data_augment == "shuf_sent":
        print('Data Augmentation Method: Shuffle Sentences')
        data_augment_method = RandomDataAugmentModule()
    else:
        print('No Data Augmentation')
        
    train_iterator = BatchingTrainFixedChunkSize(batch_size=train_params.backprop_batch_size,
                                                 dataset=dataset, model=model,
                                                 max_query_len=train_params.max_query_len,
                                                 max_doc_len=train_params.max_doc_len,
                                                 train_sampler=train_sampler,
                                                 data_augment_module=data_augment_method)

    sync_qty = 0

    for batch in train_iterator():

        with auto_cast_class():
            batch: BatchObject = batch
            batch.to(device_name)
            model_scores = model(*batch.features)
            assert len(model_scores) == len(batch)
            scores = model_scores + batch.cand_scores * cand_score_weight

            data_qty = len(batch)
            count = data_qty // train_sampler.get_chunk_size()
            assert count * train_sampler.get_chunk_size() == data_qty
            scores = scores.reshape(count, 1 + neg_qty_per_query)
            loss = loss_obj.compute(scores)

        scaler.scale(loss).backward()
        total_qty += count

        if train_params.print_grads:
            tqdm.write(f'Records processed {total_qty} Gradient sums:')
            for k, v in model.named_parameters():
                tqdm.write(k + ' ' + str('None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2))))

        total_loss += loss.item()

        # If it's time to validate, we need to interrupt the batch
        if total_qty - total_prev_qty >= batch_size:

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            total_prev_qty = total_qty

            # Scheduler must make a step in each batch! *AFTER* the optimizer makes an update!
            if scheduler is not None:
                scheduler.step()
                lr_desc = get_lr_desc(optimizer)

            # This must be done in every process, not only in the master process
            if device_qty > 1:
                if batch_id % train_params.batch_sync_qty == 0:
                    if sync_qty < sync_qty_target:
                        try:
                            sync_barrier.wait(BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT)
                        except BrokenBarrierError:
                            raise Exception('A waiting-for-model-parameter-synchronization timeout!')
                        sync_qty += 1
                        avg_model_params(model, train_params.amp)

            batch_id += 1

        if pbar is not None:
            pbar.update(count)
            pbar.refresh()
            sync_out_streams()
            pbar.set_description('%s train loss %.5f' % (lr_desc, total_loss / float(total_qty)))

    # Final model averaging in the end.

    if device_qty > 1:
        # This ensures we go through the barrier and averaging parameters exactly the same number of time in each process
        # sync_qty_target + 1 is to ensure we make at least one more final sync after the end of the epoch
        while sync_qty < sync_qty_target + 1:
            try:
                sync_barrier.wait(BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT)
            except BrokenBarrierError:
                raise Exception('A waiting-for-model-parameter-synchronization timeout!')
            sync_qty += 1
            avg_model_params(model, train_params.amp)


    #
    # Might be a bit paranoid, but this ensures no process terminates before the last avg_model_params finishes
    #
    try:
        sync_barrier.wait(BARRIER_WAIT_MODEL_AVERAGE_TIMEOUT)
    except BrokenBarrierError:
        raise Exception('A waiting-for-model-parameter-synchronization timeout!')

    if pbar is not None:
        pbar.close()
        sync_out_streams()

    return total_loss / float(total_qty)


def do_train(device_qty,
             master_port, distr_backend,
             dataset,
             qrels, qrel_file_name,
             train_pairs_all, valid_run,
             model_out_dir,
             model_holder,
             loss_obj,
             train_params):

    sync_barrier = Barrier(device_qty)

    bert_param_keys = model_holder.model.bert_param_names()

    tqdm.write('Training parameters:')
    tqdm.write(str(train_params))
    tqdm.write('BERT parameters:')
    tqdm.write(str(bert_param_keys))
    tqdm.write('Loss function:' + loss_obj.name())

    epoch_lr_decay = train_params.epoch_lr_decay

    lr = train_params.init_lr
    bert_lr = train_params.init_bert_lr

    top_valid_score = None

    train_stat = {}

    bpte = train_params.batches_per_train_epoch

    for epoch in range(train_params.epoch_qty):
        if bpte is not None and bpte >= 0:
            qty_short = int(bpte) * train_params.batch_size
            qids_all: List = list(train_pairs_all.keys())
            qids_short = np.random.choice(qids_all, qty_short, replace=False)
            train_pairs_short = {qid: train_pairs_all[qid] for qid in qids_short}
        else:
            train_pairs_short = train_pairs_all

        start_train_time = time.time()
        qids = list(train_pairs_short.keys())

        if qids:
            proc_specific_params = []
            device_name_arr = get_device_name_arr(device_qty, train_params.device_name)

            train_pair_qty = len(train_pairs_short)

            for rank in range(device_qty):
                if device_qty > 1:
                    tpart_qty = int((train_pair_qty + device_qty - 1) / device_qty)
                    train_start = rank * tpart_qty
                    train_end = min(train_start + tpart_qty, len(qids))
                    train_pairs = {k: train_pairs_short[k] for k in qids[train_start: train_end]}
                else:
                    train_pairs = train_pairs_short

                device_name = device_name_arr[rank]

                tqdm.write('Process rank %d device %s using %d training pairs out of %d' %
                           (rank, device_name, len(train_pairs), train_pair_qty))

                proc_specific_params.append(
                    {
                        'device_name': device_name,
                        'train_pairs': train_pairs
                    }
                )

            # The number of synchronization points need to be adjusted by the number of devies processes,
            # as well as by the batch size
            sync_qty_target = int(flexneuart.io.train_data.train_item_qty_upper_bound(train_pairs_short,
                                                                                      train_params.epoch_repeat_qty) / \
                                  (device_qty * train_params.batch_sync_qty * train_params.batch_size))

            shared_params = {
                'lr': lr,
                'bert_lr': bert_lr,
                'model_holder': model_holder,
                'sync_barrier': sync_barrier,
                'sync_qty_target': sync_qty_target,
                'device_qty': device_qty,
                'loss_obj': loss_obj,
                'train_params': train_params,
                'dataset': dataset,
                'qrels': qrels
            }

            loss = run_distributed(train_iteration,
                                        shared_params=shared_params,
                                        proc_specific_params=proc_specific_params,
                                        master_port=master_port,
                                        distr_backend=distr_backend,
                                        proc_qty=device_qty)
        else:
            loss = 0

        end_train_time = time.time()

        snapshot_saved = False

        if train_params.save_epoch_snapshots:
            tqdm.write('Saving the last epoch model snapshot')
            model_holder.save_all(os.path.join(model_out_dir, f'model.{epoch}'))
            snapshot_saved = True

        os.makedirs(model_out_dir, exist_ok=True)

        tqdm.write(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g}')

        sync_out_streams()

        start_val_time = time.time()

        # Run validation if the validation type is
        run_val = (train_params.valid_type == VALID_ALWAYS) or \
                  ((train_params.valid_type == VALID_LAST) and epoch + 1 == train_params.epoch_qty)

        if run_val:
            valid_score = validate(model=model_holder.model,
                                   master_port=master_port,
                                   distr_backend=distr_backend,
                                   device_qty=device_qty,
                                   train_params=train_params,
                                   dataset=dataset,
                                   orig_run=valid_run,
                                   qrelf=qrel_file_name,
                                   run_filename=os.path.join(model_out_dir, f'{epoch}.run'))
        else:
            tqdm.write(f'No validation at epoch: {epoch}')
            valid_score = None

        end_val_time = time.time()

        sync_out_streams()

        if valid_score is not None:
            tqdm.write(f'validation epoch={epoch} score={valid_score:.4g}')

        train_stat[epoch] = {'loss': loss,
                              'score': valid_score,
                              'lr': lr,
                              'bert_lr': bert_lr,
                              'train_time': end_train_time - start_train_time,
                              'validation_time': end_val_time - start_val_time}

        save_json(os.path.join(model_out_dir, 'train_stat.json'), train_stat)

        if run_val:
            if top_valid_score is None or valid_score > top_valid_score:
                top_valid_score = valid_score
                tqdm.write('new top validation score, saving the whole model')
                model_holder.save_all(os.path.join(model_out_dir, 'model.best'))
        else:
            if epoch + 1 == train_params.epoch_qty and not snapshot_saved:
                tqdm.write('Saving the last epoch snapshot')
                model_holder.save_all(os.path.join(model_out_dir, f'model.{epoch}'))

        lr *= epoch_lr_decay
        bert_lr *= epoch_lr_decay


def run_model_wrapper(model, is_main_proc, rerank_run_file_name,
                      device_name, batch_size, amp,
                      max_query_len, max_doc_len,
                      dataset, orig_run,
                      cand_score_weight,
                      desc):

    model.eval()
    model.to(device_name)

    res = run_model(model,
                      device_name, batch_size, amp,
                      max_query_len, max_doc_len,
                      dataset, orig_run,
                      cand_score_weight=cand_score_weight,
                      desc=desc, 
                      use_progress_bar=is_main_proc)
    
    write_run_dict(res, rerank_run_file_name)


def validate(model,
             device_qty,
             master_port, distr_backend,
             train_params,
             dataset,
             orig_run,
             qrelf,
             run_filename):
    """
        Model validation step:
         1. Re-rank a given run
         2. Save the re-ranked run
         3. Evaluate results

        :param model:           a model reference.
        :param device_qty:      a number of devices to use for validation
        :param master_port:     optional master port (mandatory if proc_qty > 1)
        :param distr_backend:   optional distributed backend type (mandatory if proc_qty > 1)
        :param train_params:    training parameters
        :param dataset:         validation dataset
        :param orig_run:        a run to re-rank
        :param qrelf:           QREL files
        :param run_filename:    a file name to store the *RE-RANKED* run
        :return: validation score

    """
    sync_out_streams()

    proc_specific_params = []
    device_name_arr = get_device_name_arr(device_qty, train_params.device_name)
    rerank_file_name_arr = [f'{run_filename}_{rank}' for rank in range(device_qty)]

    qids = list(orig_run.keys())
    valid_pair_qty = len(qids)

    for rank in range(device_qty):
        if device_qty > 1:
            vpart_qty = int((valid_pair_qty + device_qty - 1) / device_qty)
            valid_start = rank * vpart_qty
            valid_end = min(valid_start + vpart_qty, len(qids))
            run = {k: orig_run[k] for k in qids[valid_start: valid_end]}
        else:
            run = orig_run

        device_name = device_name_arr[rank]

        tqdm.write('Process rank %d device %s using %d validation queries out of %d' %
                    (rank, device_name, len(run), valid_pair_qty))

        proc_specific_params.append(
            {
                'device_name': device_name, 
                'orig_run': run,
                'rerank_run_file_name': rerank_file_name_arr[rank]
            }
        )

    shared_params = {
        'model': model,
        'batch_size': train_params.batch_size_val,
        'cand_score_weight': train_params.cand_score_weight,
        'amp': train_params.amp,
        'max_query_len': train_params.max_query_len,
        'max_doc_len': train_params.max_doc_len,
        'dataset': dataset,
        'desc': 'validation'
    }

    run_distributed(run_model_wrapper,
                    shared_params=shared_params,
                    proc_specific_params=proc_specific_params,
                    master_port=master_port,
                    distr_backend=distr_backend,
                    proc_qty=device_qty)

    rerank_run = {}

    for k in range(device_qty):
        sub_run = read_run_dict(rerank_file_name_arr[k])

        for k, v in sub_run.items():
            assert not k in rerank_run
            rerank_run[k] = v

    qty0 = len(rerank_run) 
    qty1 = len(orig_run)
    assert qty0 == qty1,\
           f'Something went wrong: # of queries of original and re-ranked runs re diff.: {qty0} vs {qty1}'

    eval_metric = train_params.eval_metric

    sync_out_streams()

    tqdm.write('')
    tqdm.write(f'Evaluating run with QREL file {qrelf} using metric {eval_metric}')

    sync_out_streams()

    # Let us always save the run
    return get_eval_results(use_external_eval=train_params.use_external_eval,
                              eval_metric=eval_metric,
                              rerank_run=rerank_run,
                              qrel_file=qrelf,
                              run_file=run_filename)


def main_cli():
    parser = argparse.ArgumentParser('model training and validation')

    add_model_init_basic_args(parser, add_device_name=True, add_init_model_weights=True, mult_model=False)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=flexneuart.config.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=flexneuart.config.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')

    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=str, nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=str, required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=str, required=True)

    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=str, required=True)

    parser.add_argument('--model_out_dir',
                        metavar='model out dir', help='an output directory for the trained model',
                        required=True)

    parser.add_argument('--epoch_qty', metavar='# of epochs', help='# of epochs',
                        type=int, default=10)

    parser.add_argument('--epoch_repeat_qty',
                        metavar='# of each epoch repetition',
                        help='# of times each epoch is "repeated"',
                        type=int, default=1)

    parser.add_argument('--valid_type',
                        default=VALID_ALWAYS,
                        choices=[VALID_ALWAYS, VALID_LAST, VALID_NONE],
                        help='validation type')

    parser.add_argument('--warmup_pct', metavar='warm-up fraction',
                        default=None, type=float,
                        help='use a warm-up/cool-down learning-reate schedule')
    parser.add_argument('--lr_schedule', metavar='LR schedule',
                        default=LR_SCHEDULE_CONST,
                        choices=[LR_SCHEDULE_CONST, LR_SCHEDULE_CONST_WARMUP, LR_SCHEDULE_ONE_CYCLE_LR])

    parser.add_argument('--device_qty', type=int, metavar='# of device for multi-GPU training',
                        default=1, help='# of GPUs for multi-GPU training')

    parser.add_argument('--batch_sync_qty', metavar='# of batches before model sync',
                        type=int, default=4, help='model syncronization frequency for multi-GPU trainig in the # of batche')

    parser.add_argument('--master_port', type=int, metavar='pytorch master port',
                        default=None, help='pytorch master port for multi-GPU training')

    parser.add_argument('--print_grads', action='store_true',
                        help='print gradient norms of parameters')

    parser.add_argument('--save_epoch_snapshots', action='store_true',
                        help='save model after each epoch')

    parser.add_argument('--seed', metavar='random seed', help='random seed',
                        type=int, default=42)

    parser.add_argument('--optim', metavar='optimizer', choices=[OPT_SGD, OPT_ADAMW], default=OPT_ADAMW,
                        help='Optimizer')

    parser.add_argument('--loss_margin', metavar='loss margin', help='Margin in the margin loss',
                        type=float, default=1.0)

    # If we use the listwise loss, it should be at least two negatives by default
    parser.add_argument('--neg_qty_per_query', metavar='listwise negatives',
                        help='Number of negatives per query for a listwise losse',
                        type=int, default=2)

    parser.add_argument('--init_lr', metavar='init learn. rate',
                        type=float, default=0.001, help='initial learning rate for BERT-unrelated parameters')

    parser.add_argument('--momentum', metavar='SGD momentum',
                        type=float, default=0.9, help='SGD momentum')

    parser.add_argument('--cand_score_weight', metavar='candidate provider score weight',
                        type=float, default=0.0,
                        help='a weight of the candidate generator score used to combine it with the model score.')

    parser.add_argument('--init_bert_lr', metavar='init BERT learn. rate',
                        type=float, default=0.00005, help='initial learning rate for BERT parameters')

    parser.add_argument('--epoch_lr_decay', metavar='epoch LR decay',
                        type=float, default=1.0, help='per-epoch learning rate decay')

    parser.add_argument('--weight_decay', metavar='weight decay',
                        type=float, default=0.0, help='optimizer weight decay')

    parser.add_argument('--batch_size', metavar='batch size',
                        type=int, default=32, help='batch size')

    parser.add_argument('--batch_size_val', metavar='val batch size',
                        type=int, default=32, help='validation batch size')

    parser.add_argument('--backprop_batch_size', metavar='backprop batch size',
                        type=int, default=1,
                        help='batch size for each backprop step')

    parser.add_argument('--batches_per_train_epoch', metavar='# of rand. batches per epoch',
                        type=int, default=None,
                        help='# of random batches per epoch: 0 tells to use all data')

    parser.add_argument('--max_query_val', metavar='max # of val queries',
                        type=int, default=None,
                        help='max # of validation queries')

    parser.add_argument('--no_shuffle_train', action='store_true',
                        help='disabling shuffling of training data')

    parser.add_argument('--use_external_eval', action='store_true',
                        help='use external eval tools: gdeval or trec_eval')

    parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                        help='Metric list: ' +  ','.join(METRIC_LIST), 
                        metavar='eval metric')

    parser.add_argument('--distr_backend', choices=['gloo', 'nccl'], default='gloo',
                        metavar='distr backend', help='Pytorch backend for distributed processing')

    parser.add_argument('--loss_func', choices=LOSS_FUNC_LIST,
                        default=PairwiseMarginRankingLossWrapper.name(),
                        help='Loss functions: ' + ','.join(LOSS_FUNC_LIST))

    parser.add_argument('--data_augment', metavar='Data Augmentation Method',
                        type=str, default=None,
                        help='select data augmentation method: shuf_sent')

    parser.add_argument('--json_conf', metavar='JSON config',
                        type=str, default=None,
            help='a JSON config (simple-dictionary): keys are the same as args, takes precedence over command line args')

    args = parser.parse_args()

    all_arg_names = vars(args).keys()

    if args.json_conf is not None:
        conf_file = args.json_conf
        print(f'Reading configuration variables from {conf_file}')
        add_conf = read_json(conf_file)
        for arg_name, arg_val in add_conf.items():
            arg_name : str
            if arg_name not in all_arg_names and not arg_name.startswith(MODEL_PARAM_PREF):
                print(f'Invalid option in the configuration file: {arg_name}')
                sys.exit(1)
            arg_default = getattr(args, arg_name, None)
            exp_type = type(arg_default)
            if arg_default is not None and type(arg_val) != exp_type:
                print(f'Invalid type in the configuration file: {arg_name} expected type: '+str(type(exp_type)) + f' default {arg_default}')
                sys.exit(1)
            print(f'Using {arg_name} from the config')
            setattr(args, arg_name, arg_val)

    print(args)
    print(args.data_augment)
    sync_out_streams()

    set_all_seeds(args.seed)

    loss_name = args.loss_func
    if loss_name == PairwiseSoftmaxLoss.name():
        loss_obj = PairwiseSoftmaxLoss()
    elif loss_name == RankNetLoss.name():
        loss_obj = RankNetLoss()
    elif loss_name == CrossEntropyLossWrapper.name():
        loss_obj = CrossEntropyLossWrapper()
    elif loss_name == MultiMarginRankingLossWrapper.name():
        loss_obj = MultiMarginRankingLossWrapper(margin = args.loss_margin)
    elif loss_name == PairwiseMarginRankingLossWrapper.name():
        loss_obj = PairwiseMarginRankingLossWrapper(margin = args.loss_margin)
    else:
        print('Unsupported loss: ' + loss_name)
        sys.exit(1)

    print('Loss:', loss_obj)

    # For details on our serialization approach, see comments in the ModelWrapper
    model_holder : ModelSerializer = None

    if args.init_model is not None:
        print('Loading a complete model from:', args.init_model)
        model_holder = ModelSerializer.load_all(args.init_model)
    else:
        if args.model_name is None:
            print('--model_name argument must be provided unless --init_model points to a fully serialized model!')
            sys.exit(1)
        if args.init_model_weights is not None:
            model_holder = ModelSerializer(args.model_name)
            model_holder.create_model_from_args(args)
            print('Loading model weights from:', args.init_model_weights)
            model_holder.load_weights(args.init_model_weights, strict=False)
        else:
            model_holder = ModelSerializer(args.model_name)
            print('Creating the model from scratch!')
            model_holder.create_model_from_args(args)

    if args.neg_qty_per_query < 1:
        print('A number of negatives per query cannot be < 1')
        sys.exit(1)

    os.makedirs(args.model_out_dir, exist_ok=True)
    print(model_holder.model)
    sync_out_streams()

    dataset = flexneuart.io.train_data.read_datafiles(args.datafiles)
    qrelf = args.qrels
    qrels = read_qrels_dict(qrelf)
    train_pairs_all = flexneuart.io.train_data.read_pairs_dict(args.train_pairs)
    valid_run = read_run_dict(args.valid_run)
    max_query_val = args.max_query_val
    query_ids = list(valid_run.keys())
    if max_query_val is not None:
        query_ids = query_ids[0:max_query_val]
        valid_run = {k: valid_run[k] for k in query_ids}

    print('# of eval. queries:', len(query_ids), ' in the file', args.valid_run)


    device_qty = args.device_qty

    master_port = args.master_port
    if device_qty > 1:
        # Tokenizer parallelism creates problems with multiple processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if master_port is None:
            print('Specify a master port for distributed training!')
            sys.exit(1)

    train_params = TrainParams(init_lr=args.init_lr, init_bert_lr=args.init_bert_lr,
                               device_name=args.device_name,
                               momentum=args.momentum, amp=args.amp,
                               warmup_pct=args.warmup_pct,
                               lr_schedule=args.lr_schedule,
                               batch_sync_qty=args.batch_sync_qty,
                               epoch_lr_decay=args.epoch_lr_decay, weight_decay=args.weight_decay,
                               backprop_batch_size=args.backprop_batch_size,
                               batches_per_train_epoch=args.batches_per_train_epoch,
                               save_epoch_snapshots=args.save_epoch_snapshots,
                               batch_size=args.batch_size, batch_size_val=args.batch_size_val,
                               # These lengths must come from the model serializer object, not from the arguments,
                               # because they can be overridden when the model is loaded.
                               max_query_len=model_holder.max_query_len, max_doc_len=model_holder.max_doc_len,
                               epoch_qty=args.epoch_qty, epoch_repeat_qty=args.epoch_repeat_qty,
                               cand_score_weight=args.cand_score_weight,
                               neg_qty_per_query=args.neg_qty_per_query,
                               use_external_eval=args.use_external_eval, eval_metric=args.eval_metric.lower(),
                               print_grads=args.print_grads,
                               shuffle_train=not args.no_shuffle_train,
                               valid_type=args.valid_type,
                               optim=args.optim,
                               data_augment=args.data_augment)

    do_train(
        device_qty=device_qty,
        master_port=master_port, distr_backend=args.distr_backend,
        dataset=dataset,
        qrels=qrels, qrel_file_name=qrelf,
        train_pairs_all=train_pairs_all, valid_run=valid_run,
        model_out_dir=args.model_out_dir,
        model_holder=model_holder,
        loss_obj=loss_obj,
        train_params=train_params
    )


if __name__ == '__main__':
    # A light-weight subprocessing + this is a must for multi-processing with CUDA
    enable_spawn()
    main_cli()
