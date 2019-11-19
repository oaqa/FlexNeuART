# This code is taken from CEDR: https://github.com/Georgetown-IR-Lab/cedr
# (c) Georgetown IR lab
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import os
import gc
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import modeling_duet
import data


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

BATCH_SIZE = 32
GRAD_ACC_SIZE = 16
MAX_SUBBATCH_SIZE = GRAD_ACC_SIZE
BATCH_SIZE_EVAL = GRAD_ACC_SIZE
#BATCHES_PER_TRAIN_EPOCH = 8192
#BATCHES_PER_TRAIN_EPOCH = 128
BATCHES_PER_TRAIN_EPOCH = 2048

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker,
    'duet_bert' : modeling_duet.DuetBertRanker
}


def main(model, max_epoch, no_cuda, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    LR = 0.001
    BERT_LR = 2e-5

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    for epoch in range(max_epoch):
        loss = train_iteration(model, no_cuda, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}')
        valid_score = validate(model, no_cuda, dataset, valid_run, qrelf, epoch, model_out_dir)
        print(f'validation epoch={epoch} score={valid_score}')
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights.p'))


def train_iteration(model, no_cuda, optimizer, dataset, train_pairs, qrels):
    total_prev = total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_TRAIN_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, no_cuda, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'], 
                           max_batch_size=MAX_SUBBATCH_SIZE)
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pairwise softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            gc.collect()
            if total  - total_prev >= BATCH_SIZE:
                #print(total, 'optimizer step!')
                optimizer.step()
                optimizer.zero_grad()
                total_prev = total
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_TRAIN_EPOCH:
                return total_loss


def validate(model, no_cuda, dataset, run, qrelf, epoch, model_out_dir):
    #VALIDATION_METRIC = 'P.20'
    # Leo's choice to use map
    VALIDATION_METRIC = 'map'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    run_model(model, no_cuda, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(model, no_cuda, dataset, run, runf, desc='valid'):
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, no_cuda, dataset, run, BATCH_SIZE_EVAL):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'],
                           max_batch_size=MAX_SUBBATCH_SIZE)
            gc.collect()
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')


def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'trec_eval/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+', required=True)
    parser.add_argument('--qrels', type=argparse.FileType('rt'), required=True)
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'), required=True)
    parser.add_argument('--valid_run', type=argparse.FileType('rt'), required=True)
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'), default=None)
    parser.add_argument('--model_out_dir', required=True)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    model = MODEL_MAP[args.model]()
    if not args.no_cuda:
        model = model.cuda()
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, args.max_epoch, args.no_cuda, dataset, train_pairs, qrels, valid_run, qrelf=args.qrels.name, model_out_dir=args.model_out_dir)


if __name__ == '__main__':
    main_cli()
