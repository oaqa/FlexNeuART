#!/usr/bin/env python
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
# It has some substantial modifications + it relies on our custom BERT
# library: https://github.com/searchivarius/pytorch-pretrained-BERT-mod
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import os
import sys
import argparse
import subprocess

sys.path.append('scripts')

import torch
import modeling
import modeling_dssm
import utils
import data

from matchzoo_metrics.mean_average_precision import MeanAveragePrecision
from matchzoo_metrics.discounted_cumulative_gain import DiscountedCumulativeGain

from common_eval import *

METRIC_MAP='map'
MAP_TOP_K=10000
METRIC_NDCG='ndcg@20'
NDCG_TOP_K=20

METRIC_LIST=[METRIC_MAP, METRIC_NDCG]

from tqdm import tqdm
from collections import namedtuple

class MarginRankingLossWrapper:
  @staticmethod
  def name():
    return 'pairwise_margin'

  '''This is a wrapper class for the margin ranking loss.
     It expects that positive/negative scores are arranged in pairs'''
  def __init__(self, margin):
    self.loss = torch.nn.MarginRankingLoss(margin)

  def compute(self, scores):
    pos_doc_scores = scores[:, 0]
    neg_doc_scores = scores[:, 1]
    ones = torch.ones_like(pos_doc_scores)
    return self.loss.forward(pos_doc_scores, neg_doc_scores, target=ones)


class PairwiseSoftmaxLoss:
  @staticmethod
  def name():
    return 'pairwise_softmax'

  '''This is a wrapper class for the pairwise softmax ranking loss.
     It expects that positive/negative scores are arranged in pairs'''
  def compute(self, scores):
    return torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pairwise softmax

LOSS_FUNC_LIST=[PairwiseSoftmaxLoss.name(), MarginRankingLossWrapper.name()]

TrainParams = namedtuple('TrainParams',
                    ['init_lr', 'init_bert_lr', 'epoch_lr_decay',
                     'batches_per_train_epoch',
                     'batch_size', 'batch_size_eval',
                     'max_query_len', 'max_doc_len',
                     'backprop_batch_size',
                     'epoch_qty', 'no_cuda', 'print_grads',
                     'shuffle_train',
                     'use_external_eval', 'eval_metric'])

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker,
    'dssm_bert' : modeling_dssm.DssmBertRanker
}


def main(model, loss_obj, train_params, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    lr = train_params.init_lr
    bert_lr = train_params.init_bert_lr
    decay = train_params.epoch_lr_decay

    top_valid_score = None

    for epoch in range(train_params.epoch_qty):

        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
        bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': bert_lr}

        optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=lr)
        loss = train_iteration(model, loss_obj, train_params, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss:.3g} lr={lr:g} bert_lr={bert_lr:g}')
        valid_score = validate(model, train_params, dataset, valid_run, qrelf, epoch, model_out_dir)
        print(f'validation epoch={epoch} score={valid_score:.4g}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))

        lr *= decay
        bert_lr *= decay


def train_iteration(model, loss_obj, train_params, optimizer, dataset, train_pairs, qrels):

    model.train()
    total_loss = 0.
    total_prev_qty = total_qty = 0. # This is a total number of records processed, it can be different from
                                    # the total number of training pairs

    bpte = train_params.batches_per_train_epoch
    batch_size = train_params.batch_size
    max_train_qty = data.train_item_qty(train_pairs) if bpte <= 0 else bpte * batch_size

    optimizer.zero_grad()

    if train_params.print_grads:
      print('Gradient sums before training')
      for k, v in model.named_parameters():
        print(k, 'None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2)))

    with tqdm('training', total=max_train_qty, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, train_params.no_cuda, dataset, train_pairs, train_params.shuffle_train,
                                            qrels, train_params.backprop_batch_size,
                                            train_params.max_query_len, train_params.max_doc_len):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = loss_obj.compute(scores)
            loss.backward()
            total_qty += count

            if train_params.print_grads:
              print(f'Records processed {total_qty} Gradient sums:')
              for k, v in model.named_parameters():
                print(k, 'None' if v.grad is None else torch.sum(torch.norm(v.grad, dim=-1, p=2)))

            total_loss += loss.item()

            if total_qty  - total_prev_qty >= batch_size:
                #print(total, 'optimizer step!')
                optimizer.step()
                optimizer.zero_grad()
                total_prev_qty = total_qty
            pbar.update(count)
            if total_qty >= max_train_qty:
                break

    return total_loss / float(total_qty)


def validate(model, train_params, dataset, run, qrelf, epoch, model_out_dir):
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    run_model(model, train_params, dataset, run, runf)
    eval_metric = train_params.eval_metric
    print(f'Evaluating run {runf} with QREL file {qrelf} using metric {eval_metric}')
    if train_params.use_external_eval:
      if eval_metric == METRIC_MAP:
        VALIDATION_METRIC = 'map'
        return trec_eval(qrelf, runf, 'map')
      elif train_params.eval_metric == METRIC_NDCG:
        return gdeval(qrelf, runf, 'ndcg')
      else:
        raise Exception('Unsupported metric: ' + train_params.eval_metric)
    else:
      if eval_metric == METRIC_MAP:
        return evalRun(runf, qrelf, MeanAveragePrecision(), MAP_TOP_K, True, debug=False)
      elif train_params.eval_metric == METRIC_NDCG:
        return evalRun(runf, qrelf, DiscountedCumulativeGain(), NDCG_TOP_K, True, debug=False)
      else:
        raise Exception('Unsupported metric: ' + train_params.eval_metric)


def run_model(model, train_params, dataset, run, runf, desc='valid'):
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model,
                                               train_params.no_cuda,
                                               dataset, run,
                                               train_params.batch_size_eval,
                                               train_params.max_query_len, train_params.max_doc_len):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(genRunEntryStr(qid, did, i+1, score, FAKE_RUN_ID) + '\n')

def gdeval(qrelf, runf, metric):
    gval_f = 'scripts/exper/gdeval.pl'
    output = subprocess.check_output([gval_f, qrelf, runf]).decode().rstrip()
    output = output.split('\n')
    last = output[-1].split(',')
    metric = metric.lower()
    if metric == 'err':
      return float(last[-1])
    elif metric == 'ndcg':
      return float(last[-2])
    else:
      raise Exception('Invalid gdeval metric: ' + metric)

import multiprocessing
from multiprocessing import Process, Queue

def trec_eval_call(q, qrelf, runf, metric):
    trec_eval_f = 'trec_eval/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    q.put(float(output[0].split()[2]))

def trec_eval(qrelf, runf, metric):
    q = Queue()
    # Before calling check_output we need to start a light-weight version
    # of the child process, otherwise we could run out of memory
    p = Process(target=trec_eval_call, args=(q, qrelf, runf, metric))
    p.start()
    res = q.get()
    p.join()
    return res


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', metavar='model', help='a model to use',
                        choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=argparse.FileType('rt'), nargs='+', required=True)
    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=argparse.FileType('rt'), required=True)
    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=argparse.FileType('rt'), required=True)
    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=argparse.FileType('rt'), required=True)
    parser.add_argument('--initial_bert_weights',
                        metavar='initial BERT weights', help='initial BERT weights',
                        type=argparse.FileType('rb'), default=None)
    parser.add_argument('--model_out_dir',
                        metavar='model out dir', help='an output directory for the trained model',
                        required=True)
    parser.add_argument('--epoch_qty', metavar='# of epochs', help='# of epochs',
                        type=int, default=10)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--print_grads', action='store_true')
    parser.add_argument('--seed', metavar='random seed', help='random seed',
                        type=int, default=42)
    parser.add_argument('--loss_margin', metavar='loss margin', help='Margin in the margin loss',
                        type=float, default=1)
    parser.add_argument('--init_lr', metavar='init learn. rate',
                        type=float, default=0.001, help='Initial learning rate for BERT-unrelated parameters')
    parser.add_argument('--init_bert_lr', metavar='init BERT learn. rate',
                        type=float, default=0.00005, help='Initial learning rate for BERT parameters')
    parser.add_argument('--epoch_lr_decay', metavar='epoch LR decay',
                        type=float, default=0.9, help='Per-epoch learning rate decay')
    parser.add_argument('--batch_size', metavar='batch size',
                        type=int, default=32, help='batch size')
    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=32, help='max. query length')
    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=2048,
                        help='max. document length')
    parser.add_argument('--batch_size_eval', metavar='eval batch size',
                        type=int, default=64, help='batch size for evaluation')
    parser.add_argument('--backprop_batch_size', metavar='backprop batch size',
                        type=int, default=12,
                        help='batch size for each backprop step')
    parser.add_argument('--batches_per_train_epoch', metavar='# of rand. batches per epoch',
                        type=int, default=0,
                        help='# of random batches per epoch: 0 tells to use all data')
    parser.add_argument('--max_query_eval', metavar='max # of eval queries',
                        type=int, default=0,
                        help='max # of evaluation queries: 0 tells to use all data')
    parser.add_argument('--grad_checkpoint_param', type=int, default=0,
                        metavar='grad. checkpoint param',
                       help='gradient checkpointing param (0, no checkpointing, 2 every other layer, 3 every 3rd layer, ...)')
    parser.add_argument('--no_shuffle_train', action='store_true',
                        help='disabling shuffling of training data')
    parser.add_argument('--use_external_eval', action='store_true',
                        help='use external eval tools: gdeval or trec_eval')
    parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                        help='Metric list: ' +  ','.join(METRIC_LIST), 
                        metavar='eval metric')
    parser.add_argument('--loss_func', choices=LOSS_FUNC_LIST,
                        default=PairwiseSoftmaxLoss.name(),
                        help='Loss functions: ' +  ','.join(LOSS_FUNC_LIST))
    args = parser.parse_args()

    utils.set_all_seeds(args.seed)

    loss_name = args.loss_func
    if loss_name == PairwiseSoftmaxLoss.name():
      loss_obj = PairwiseSoftmaxLoss()
    elif loss_name == MarginRankingLossWrapper.name():
      loss_obj = MarginRankingLossWrapper(margin = args.loss_margin)
    else:
      raise Exception('Unsupported loss: ' + loss_name)

    train_params = TrainParams(init_lr=args.init_lr, init_bert_lr=args.init_bert_lr, epoch_lr_decay=args.epoch_lr_decay,
                         backprop_batch_size=args.backprop_batch_size,
                         batches_per_train_epoch=args.batches_per_train_epoch,
                         batch_size=args.batch_size, batch_size_eval=args.batch_size_eval,
                         max_query_len=args.max_query_len, max_doc_len=args.max_doc_len,
                         epoch_qty=args.epoch_qty, no_cuda=args.no_cuda,
                         use_external_eval=args.use_external_eval, eval_metric=args.eval_metric.lower(),
                         print_grads=args.print_grads,
                         shuffle_train=not args.no_shuffle_train)

    print('Training parameters:')
    print(train_params)
    print('Loss function:', loss_obj.name())

    model = MODEL_MAP[args.model]()
    model.set_grad_checkpoint_param(args.grad_checkpoint_param)
    if not args.no_cuda:
        model = model.cuda()
    dataset = data.read_datafiles(args.datafiles)
    qrels = readQrelsDict(args.qrels.name)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = readRunDict(args.valid_run.name)
    max_query_eval=args.max_query_eval
    query_ids = list(valid_run.keys())
    if max_query_eval > 0:
      valid_run = {k:valid_run[k] for k in query_ids[0:max_query_eval]}
    print('# of eval. queries:', len(query_ids), ' in the file', args.valid_run.name)
      
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, loss_obj, train_params,
         dataset, train_pairs, qrels, valid_run, qrelf=args.qrels.name, model_out_dir=args.model_out_dir)


if __name__ == '__main__':
    # Hopefully a light-weight subprocessing
    utils.enable_spawn()
    main_cli()
