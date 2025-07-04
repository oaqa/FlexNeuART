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

import os
import time
import sys
import math
import argparse
import json
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers.optimization import get_constant_schedule_with_warmup
from typing import List, Tuple, Dict, Any

import flexneuart.config
import flexneuart.io.train_data

from flexneuart.models.utils import add_model_init_basic_args

from flexneuart.models.train import run_model, clean_memory
from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.models.train.batch_obj import BatchObject
from flexneuart.models.train.batching import TrainSamplerFixedChunkSize,\
                                             BatchingTrainFixedChunkSize

from flexneuart.models.train.distr_utils import run_distributed, get_device_name_arr, \
                                                enable_spawn, avg_model_params
from flexneuart.models.train.loss import *
from flexneuart.models.train.amp import get_amp_processors

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
                     'init_lr', 'init_bert_lr', 
                     'init_aggreg_lr', 'init_bart_lr',
                     'init_interact_lr',
                     'epoch_lr_decay', 'weight_decay',
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

class FlexNeuArtDataset(Dataset):
    def __init__(self, queries, docs, neg_qty_per_query, epoch_repeat_qty, max_query_length, max_doc_length, query_groups):
        self.queries = queries
        self.docs = docs
        self.neg_qty_per_query = neg_qty_per_query
        self.epoch_repeat_qty = epoch_repeat_qty
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.query_groups = query_groups
        self.group_size = (1 + self.neg_qty_per_query)
        self.epoch_example_count = len(self.query_groups) * self.group_size
        self.train_example_count = self.epoch_example_count * epoch_repeat_qty
        self.qidx = 0
    
    def __len__(self):
        return self.train_example_count

    def __getitem__(self, idx):
        example_idx = idx % self.epoch_example_count
        
        group_idx = example_idx // self.group_size
        instance_idx = example_idx % self.group_size

        group = self.query_groups[group_idx]

        qid = group['qid']
        
        if instance_idx == 0:
            docid, cand_score = group['pos_id'], group['pos_id_score']
        else:
            docid, cand_score = group['neg_ids'][instance_idx - 1], group['neg_ids_score'][instance_idx - 1]

        query_text = self.queries[qid]
        doc_text = self.docs[docid]

        label = 1 if instance_idx == 0 else 0

        return qid, docid, label, cand_score, query_text, doc_text

def create_collate_fn(model, max_query_length, max_doc_length):
    def collate_fn(batch):
        qids, docids, labels, cand_scores, query_text, doc_text = zip(*batch)

        features = model.featurize(max_query_len=max_query_length,
                                   max_doc_len=max_doc_length,
                                   query_texts=query_text,
                                   doc_texts=doc_text)

        return qids, docids, labels, torch.FloatTensor(cand_scores), features
    return collate_fn


def train_epoch(accelerator: Accelerator,
                train_params: TrainParams,
                model_holder: ModelSerializer,
                dataloader: Any,
                lr: float,
                bert_lr: float,
                aggreg_lr: float,
                interact_lr: float,
                sync_qty_target: int,
                loss_obj: Any,
                wandb_run: Any):
    
    model = model_holder.model.module if hasattr(model_holder.model, "module") else model_holder.model

    if train_params.init_bart_lr is not None:
        bart_param_keys = model.bart_param_names()
    else:
        bert_param_keys = model.bert_param_names()
    
    if aggreg_lr is not None:
        aggreg_keys = model.aggreg_param_names()
    else:
        aggreg_keys = []

    if interact_lr is not None:
        interact_keys = model.interact_param_names()
    else:
        interact_keys = []

    all_params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]

    # BERT parameters use a special learning weight
    if train_params.init_bart_lr is not None:
        bart_params = {'params': [v for k, v in all_params if k in bart_param_keys], 'lr': bert_lr}
    else:
        bert_params = {'params': [v for k, v in all_params if k in bert_param_keys], 'lr': bert_lr}
    
    if aggreg_lr is not None:
        aggreg_params = {'params': [v for k, v in all_params if k in aggreg_keys], 'lr': aggreg_lr}
    else:
        aggreg_params = {}
    if  interact_lr is not None:
        interact_params = {'params': [v for k, v in all_params if k in interact_keys], 'lr': interact_lr}
    else:
        interact_params = {}
    if train_params.init_bart_lr is not None:
        non_bert_params = {'params': [v for k, v in all_params if not k in bart_param_keys and not k in aggreg_keys \
                                      and not k in interact_keys]}
    else:
        non_bert_params = {'params': [v for k, v in all_params if not k in bert_param_keys and not k in aggreg_keys \
                                      and not k in interact_keys]}
    
    if train_params.init_bart_lr is not None:
        params = list(filter(None, [non_bert_params, bart_params, interact_params, aggreg_params]))
    else:
        params = list(filter(None, [non_bert_params, bert_params, interact_params, aggreg_params]))

    if train_params.optim == OPT_ADAMW:
        optimizer = torch.optim.AdamW(params,
                                      lr=lr, weight_decay=train_params.weight_decay)
    elif train_params.optim == OPT_SGD:
        optimizer = torch.optim.SGD(params,
                                    lr=lr, weight_decay=train_params.weight_decay,
                                    momentum=train_params.momentum)
    else:
        raise Exception('Unsupported optimizer: ' + train_params.optim)

    lr_steps = len(dataloader) // (1 + train_params.neg_qty_per_query)
 
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
            
            accelerator.print(f'Using a one-cycle scheduler with a warm-up for {num_warmup_steps} steps')
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            total_steps=lr_steps,
                                                            max_lr=[lr, bert_lr, interact_lr, aggreg_lr],
                                                            anneal_strategy='linear',
                                                            pct_start=train_params.warmup_pct)
        else:
            assert lr_schedule == LR_SCHEDULE_CONST_WARMUP
            accelerator.print(f'Using a const-learning rate scheduler with a warm-up for {num_warmup_steps} steps')

            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    else:
        raise Exception('Unsupported LR schedule: ' + lr_schedule)

    if accelerator.is_main_process:
        tqdm.write('Optimizer:' + str( optimizer))

    # Freeing unused memory
    accelerator.free_memory()

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    total_loss = 0.
    total_prev_qty = total_qty = 0. # This is a total number of records processed, it can be different from
                                    # the total number of training pairs

    batch_size = train_params.batch_size

    optimizer.zero_grad()

    lr_desc = get_lr_desc(optimizer)

    batch_id = 0

    if accelerator.is_main_process:

        if train_params.print_grads:
            tqdm.write('Gradient sums before training')
            for k, v in model.named_parameters():
                tqdm.write(k + ' ' + str('None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2))))

        pbar = tqdm('training', total=lr_steps * train_params.batch_size * accelerator.num_processes, ncols=80, desc=None, leave=False, file=TQDM_FILE)
    else:
        pbar = None

    neg_qty_per_query = train_params.neg_qty_per_query
    
    cand_score_weight = torch.FloatTensor([train_params.cand_score_weight]).to(accelerator.device)
    
    auto_cast_class, scaler = get_amp_processors(train_params.amp)

    sync_qty = 0

    for batch in dataloader:
        with accelerator.accumulate(model):
            with auto_cast_class():
                model_scores = model(*batch[4])
                assert len(model_scores) == len(batch[3])
                scores = model_scores + batch[3] * cand_score_weight

                data_qty = len(batch[3])
                count = data_qty // (1 + neg_qty_per_query)
                assert count * (1 + neg_qty_per_query) == data_qty
                scores = scores.reshape(count, 1 + neg_qty_per_query)
                loss = loss_obj.compute(scores)
                if wandb_run is not None:
                    wandb_run.log({'loss': loss.item()})
        
            accelerator.backward(scaler.scale(loss))
            total_qty += count

            if accelerator.is_main_process and train_params.print_grads:
                tqdm.write(f'Records processed {total_qty * accelerator.state.num_processes} Gradient sums:')
                for k, v in model.named_parameters():
                    tqdm.write(k + ' ' + str('None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2))))

            total_loss += loss

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

                batch_id += 1

        avg_loss = accelerator.gather(total_loss).mean().item()  / float(max(1, total_qty))

        if pbar is not None:
            pbar.update(count * accelerator.num_processes)
            pbar.refresh()
            pbar.set_description('%s train loss %.5f' % (lr_desc, avg_loss))

        if wandb_run is not None:
            wandb_run.log({'avg_loss': avg_loss})


    # Final model averaging in the end.

    if accelerator.state.num_processes > 1:
        # This ensures we go through the barrier and averaging parameters exactly the same number of time in each process
        # sync_qty_target + 1 is to ensure we make at least one more final sync after the end of the epoch
        while sync_qty < sync_qty_target + 1:
            accelerator.wait_for_everyone()
            sync_qty += 1
            avg_model_params(model, train_params.amp)

    if pbar is not None:
        pbar.close()

    return accelerator.gather(total_loss).mean().item() / float(max(total_qty, 1))
        
def run_model_wrapper(model, is_main_proc, rerank_run_file_name,
                      device_name, batch_size, amp,
                      max_query_len, max_doc_len,
                      dataset, orig_run,
                      cand_score_weight,
                      desc):

    model.eval()

    res = run_model(model,
                      device_name, batch_size, amp,
                      max_query_len, max_doc_len,
                      dataset, orig_run,
                      cand_score_weight=cand_score_weight,
                      desc=desc, 
                      use_progress_bar=is_main_proc)
    
    write_run_dict(res, rerank_run_file_name)


def validate(accelerator: Accelerator,
             model: Any,
             train_params: TrainParams,
             dataset: Tuple[Dict[str, str], Dict[str, str]],
             orig_run: Dict[str, Dict[str, float]],
             qrelf: str,
             run_filename: Any):

    device_qty = accelerator.state.num_processes
    device_name_arr = get_device_name_arr(device_qty, accelerator.device)
    rerank_file_name_arr = [f'{run_filename}_{rank}' for rank in range(device_qty)]
    qids = list(orig_run.keys())
    valid_pair_qty = len(qids)

    rank = accelerator.state.process_index
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
    
    run_model_wrapper(model=model,
                      is_main_proc=accelerator.is_main_process,
                      rerank_run_file_name=rerank_file_name_arr[rank],
                      device_name=accelerator.device,
                      batch_size=train_params.batch_size,
                      amp=train_params.amp,
                      max_query_len=train_params.max_query_len,
                      max_doc_len=train_params.max_doc_len,
                      dataset=dataset,
                      orig_run=run,
                      cand_score_weight=train_params.cand_score_weight,
                      desc='validation')
    
    # Let all processes finish their piece of work
    accelerator.wait_for_everyone() 
    
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

    if accelerator.is_main_process:
        tqdm.write('')
        tqdm.write(f'Evaluating run with QREL file {qrelf} using metric {eval_metric}')

     # Let us always save the run
    write_run_dict(rerank_run, run_filename)

    return get_eval_results(use_external_eval=train_params.use_external_eval,
                            eval_metric=eval_metric,
                            run=rerank_run,
                            qrels=qrelf)

def do_train(accelerator: Accelerator,
             train_params: TrainParams,
             model_holder: ModelSerializer,
             dataset: Tuple[Dict[str, str], Dict[str, str]],
             query_groups: List,
             qrel_file_name: str,
             loss_obj: Any,
             model_out_dir: str,
             valid_run: Dict[str, Dict[str, float]],
             wandb_run: Any):
    if train_params.init_bart_lr is not None:
        bart_param_keys = model_holder.model.bart_param_names()
        bert_lr = train_params.init_bart_lr
    else:
        bert_param_keys = model_holder.model.bert_param_names()
        bert_lr = train_params.init_bert_lr

    if train_params.init_aggreg_lr is not None:
        aggregator_param_keys = model_holder.model.aggreg_param_names()
    else:
        aggregator_param_keys = []

    if train_params.init_interact_lr is not None:
        interaction_param_keys = model_holder.model.interact_param_names()
    else:
        interaction_param_keys = []
    
    if accelerator.is_main_process:
        tqdm.write('Training parameters:')
        tqdm.write(str(train_params))
        if train_params.init_bart_lr is not None:
            tqdm.write('BART parameters:')
            tqdm.write(str(bart_param_keys))
        else:
            tqdm.write('BERT parameters:')
            tqdm.write(str(bert_param_keys))
        if train_params.init_aggreg_lr is not None:
            tqdm.write('Aggregator parameters:')
            tqdm.write(str(aggregator_param_keys))
        if train_params.init_interact_lr is not None:
            tqdm.write('Interaction parameters:')
            tqdm.write(str(interaction_param_keys))
        tqdm.write('Loss function:' + loss_obj.name())

    epoch_lr_decay = train_params.epoch_lr_decay

    lr = train_params.init_lr
    
    aggreg_lr = train_params.init_aggreg_lr

    interact_lr = train_params.init_interact_lr

    top_valid_score = None

    train_stat = {}

    device_qty = accelerator.state.num_processes

    flexneuart_dataset = FlexNeuArtDataset(queries=dataset[0], 
                                           docs=dataset[1], 
                                           neg_qty_per_query=train_params.neg_qty_per_query,
                                           epoch_repeat_qty=train_params.epoch_repeat_qty,
                                           max_query_length=train_params.max_query_len,
                                           max_doc_length=train_params.max_doc_len,
                                           query_groups=query_groups)
    
    # Forcing the dataloader to emit batches that are multiple of (1+neg_qty)
    # dataloader_batch_size = int(train_params.batch_size * (1 + train_params.neg_qty_per_query))
    dataloader_batch_size = train_params.batch_size
    
    dataloader = DataLoader(flexneuart_dataset,
                            batch_size=dataloader_batch_size,
                            shuffle= False,
                            collate_fn=create_collate_fn(model_holder.model, train_params.max_query_len, train_params.max_doc_len))

    # Prepare for training
    model_holder.model, dataloader = accelerator.prepare(model_holder.model, dataloader)

    for epoch in range(train_params.epoch_qty):
        start_train_time = time.time()

        # The number of synchronization points need to be adjusted by the number of devies processes,
        # as well as by the batch size
        sync_qty_target = int(len(flexneuart_dataset) * train_params.epoch_repeat_qty) / \
                              (device_qty * train_params.batch_sync_qty * train_params.batch_size)
        
        # Training mode automatically syncs the model for next epoch
        model_holder.model.train()

        if len(flexneuart_dataset) > 0:
            loss = train_epoch(accelerator=accelerator,
                        train_params=train_params,
                        model_holder=model_holder,
                        dataloader=dataloader,
                        lr=lr,
                        bert_lr=bert_lr,
                        aggreg_lr=aggreg_lr,
                        interact_lr=interact_lr,
                        sync_qty_target=sync_qty_target,
                        loss_obj=loss_obj,
                        wandb_run=wandb_run)
        else:
            loss = 0

        end_train_time = time.time()

        snapshot_saved = False

        if train_params.save_epoch_snapshots:
            if accelerator.is_main_process:
                tqdm.write('Saving the last epoch model snapshot')
                model_holder.save_all(os.path.join(model_out_dir, f'model.{epoch}'))
                snapshot_saved = True
                os.makedirs(model_out_dir, exist_ok=True)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if aggreg_lr is not None and interact_lr is None:
                tqdm.write(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g} aggreg_lr={aggreg_lr:g}')
            elif aggreg_lr is not None and interact_lr is not None:
                tqdm.write(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g} \
                            interact_lr={interact_lr:g} aggreg_lr={aggreg_lr:g}')
            else:
                tqdm.write(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g}')
        
        start_val_time = time.time()

        # Run validation if the validation type is
        run_val = (train_params.valid_type == VALID_ALWAYS) or \
                  ((train_params.valid_type == VALID_LAST) and epoch + 1 == train_params.epoch_qty)
        
        # Changing the mode acts as a barrier to let all the processes catch up to this point.
        model_holder.model.eval()

        if run_val:
            valid_score = validate(accelerator=accelerator,
                                   model=model_holder.model.module if hasattr(model_holder.model, "module") else model_holder.model,
                                   train_params=train_params,
                                   dataset=dataset,
                                   orig_run=valid_run,
                                   qrelf=qrel_file_name,
                                   run_filename=os.path.join(model_out_dir, f'{epoch}.run'))
        else:
            tqdm.write(f'No validation at epoch: {epoch}')
            valid_score = None

        end_val_time = time.time()

        if valid_score is not None:
            tqdm.write(f'validation epoch={epoch} score={valid_score:.4g}')

        accelerator.wait_for_everyone()

        train_time = end_train_time - start_train_time
        val_time = end_val_time - start_val_time

        if accelerator.is_main_process and wandb_run is not None:
            wandb_run.log({'final_loss': loss, 'epoch': epoch, 
                           'valid_score': valid_score,
                           'train_time': train_time,
                           'validation_time': val_time})

        train_stat[epoch] = {'loss': loss,
                              'score': valid_score,
                              'metric_name': train_params.eval_metric,
                              'lr': lr,
                              'bert_lr': bert_lr,
                              'aggreg_lr': aggreg_lr,
                              'interact_lr': interact_lr,
                              'train_time': train_time,
                              'validation_time': val_time}

        if accelerator.is_main_process:
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

        accelerator.wait_for_everyone()

        lr *= epoch_lr_decay
        bert_lr *= epoch_lr_decay
        if aggreg_lr is not None:
            aggreg_lr *= epoch_lr_decay
        if interact_lr is not None:
            interact_lr *= epoch_lr_decay
    
    
def main_cli():
    parser = argparse.ArgumentParser('model training and validation')

    add_model_init_basic_args(parser, add_device_name=False, add_init_model_weights=True, mult_model=False)

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
    
    parser.add_argument('--sbert_train_file', metavar='sentence bert exported data', help='train file',
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
                        type=float, default=None, help='initial learning rate for BERT parameters')
    
    parser.add_argument('--init_bart_lr', metavar='init BART learn. rate',
                        type=float, default=None, help='initial learning rate for BERT parameters')
    
    parser.add_argument('--init_aggreg_lr', metavar='init aggregation learn. rate',
                        type=float, default=None, help='initial learning rate for aggregation parameters')
    
    parser.add_argument('--init_interact_lr', metavar='init interaction learn. rate',
                        type=float, default=None, help='initial learning rate for interaction parameters')

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
                        help='use external eval tool trec_eval')

    parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                        help='Metric list: ' +  ','.join(METRIC_LIST), 
                        metavar='eval metric')

    parser.add_argument('--distr_backend', choices=['gloo', 'nccl'], default='gloo',
                        metavar='distr backend', help='Pytorch backend for distributed processing')

    parser.add_argument('--loss_func', choices=LOSS_FUNC_LIST,
                        default=PairwiseMarginRankingLossWrapper.name(),
                        help='Loss functions: ' + ','.join(LOSS_FUNC_LIST))
    
    parser.add_argument('--wandb_api_key', metavar='wandb_api_key', type=str, default=None,
                        help='wandb api key for logging')
    
    parser.add_argument('--wandb_project', metavar='wandb_project', type=str, default=None,
                        help='wandb project for logging')
    
    parser.add_argument('--wandb_team_name', metavar='wandb_team_name', type=str, default=None,
                        help='wandb team name for logging')
    
    parser.add_argument('--wandb_run_name', metavar='wandb_run_name', type=str, default=None,
                        help='wandb run name for logging')
    
    parser.add_argument('--wandb_group_name', metavar='wandb_group_name', type=str, default=None,
                        help='wandb group name for logging')

    parser.add_argument('--json_conf', metavar='JSON config',
                        type=str, default=None,
            help='a JSON config (simple-dictionary): keys are the same as args, takes precedence over command line args')
    
    args = parser.parse_args()

    all_arg_names = vars(args).keys()

    # Creating the accelerator object to log only on the main process
    accelerator = Accelerator(gradient_accumulation_steps=args.batch_sync_qty)

    if args.json_conf is not None:
        conf_file = args.json_conf
        accelerator.print(f'Reading configuration variables from {conf_file}')
        add_conf = read_json(conf_file)
        for arg_name, arg_val in add_conf.items():
            arg_name : str
            if arg_name not in all_arg_names and not arg_name.startswith(MODEL_PARAM_PREF):
                accelerator.print(f'Invalid option in the configuration file: {arg_name}')
                sys.exit(1)
            arg_default = getattr(args, arg_name, None)
            exp_type = type(arg_default)
            if arg_default is not None and type(arg_val) != exp_type:
                accelerator.print(f'Invalid type in the configuration file: {arg_name} expected type: '+str(type(exp_type)) + f' default {arg_default}')
                sys.exit(1)
            accelerator.print(f'Using {arg_name} from the config')
            setattr(args, arg_name, arg_val)

    accelerator.print(args)
    if args.init_bert_lr is None and args.init_bart_lr is None:
        raise ValueError("At least one of init_bert_lr or init_bart_lr is required")

    set_all_seeds(args.seed)

    # Setting the same seed for accelerate
    set_seed(args.seed)

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
        accelerator.print('Unsupported loss: ' + loss_name)
        sys.exit(1)

    accelerator.print('Loss:', loss_obj)

    # For details on our serialization approach, see comments in the ModelWrapper
    model_holder : ModelSerializer = None

    if args.init_model is not None:
        accelerator.print('Loading a complete model from:', args.init_model)
        model_holder = ModelSerializer.load_all(args.init_model)
    else:
        if args.model_name is None:
            accelerator.print('--model_name argument must be provided unless --init_model points to a fully serialized model!')
            sys.exit(1)
        if args.init_model_weights is not None:
            model_holder = ModelSerializer(args.model_name)
            model_holder.create_model_from_args(args)
            accelerator.print('Loading model weights from:', args.init_model_weights)
            model_holder.load_weights(args.init_model_weights, strict=False)
        else:
            model_holder = ModelSerializer(args.model_name)
            accelerator.print('Creating the model from scratch!')
            model_holder.create_model_from_args(args)

    if args.neg_qty_per_query < 1:
        accelerator.print('A number of negatives per query cannot be < 1')
        sys.exit(1)

    if accelerator.is_main_process:
        os.makedirs(args.model_out_dir, exist_ok=True)
    
    accelerator.print(model_holder.model)

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

    accelerator.print('# of eval. queries:', len(query_ids), ' in the file', args.valid_run)

    query_groups = []
    with open(args.sbert_train_file, 'r') as sdata:
        for group in sdata:
            query_groups.append(json.loads(group))
    accelerator.print('# of query groups :', len(query_groups), ' in the file', args.sbert_train_file)

    device_qty = accelerator.state.num_processes

    if device_qty > 1:
        # Tokenizer parallelism creates problems with multiple processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if accelerator.is_main_process and args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)

        args_dict = vars(args)

        # remove any key that start with "wandb"
        config = {k: v for k, v in args_dict.items() if not k.startswith('wandb')}

        if args.wandb_project is None:
            wandb_project = args.model_name

        wandb_run = wandb.init(project=wandb_project, 
                   entity=args.wandb_team_name, 
                   config=config,
                   name=args.wandb_run_name,
                   group=args.wandb_group_name)
    else:
        wandb_run = None
    
    if  loss_obj.has_mult_negatives():
        assert args.neg_qty_per_query > 1
        neg_qty_per_query = args.neg_qty_per_query
    else:
        neg_qty_per_query = 1
    
 
    # adjusting the batch size to be a multiple of group size
    args.batch_size = int(math.ceil(args.batch_size / (1 + neg_qty_per_query)) * (1 + neg_qty_per_query))
    
    train_params = TrainParams(init_lr=args.init_lr, init_bert_lr=args.init_bert_lr,
                               init_aggreg_lr=args.init_aggreg_lr, init_bart_lr=args.init_bart_lr,
                               init_interact_lr=args.init_interact_lr,
                               device_name=accelerator.device,
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
                               neg_qty_per_query=neg_qty_per_query,
                               use_external_eval=args.use_external_eval, eval_metric=args.eval_metric.lower(),
                               print_grads=args.print_grads,
                               shuffle_train=not args.no_shuffle_train,
                               valid_type=args.valid_type,
                               optim=args.optim)
    
    do_train(accelerator=accelerator,
             train_params=train_params,
             model_holder=model_holder, 
             dataset=dataset, 
             query_groups=query_groups, 
             qrel_file_name=qrelf, 
             loss_obj=loss_obj, 
             model_out_dir=args.model_out_dir,
             valid_run=valid_run,
             wandb_run=wandb_run)
    
    # Clean up before termination
    accelerator.end_training()


if __name__ == '__main__':
    main_cli()
