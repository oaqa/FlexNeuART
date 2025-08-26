#!/usr/bin/env python

import argparse
import os
import json



#from flexneuart.io.json import read_json, save_json
import flexneuart.config
import flexneuart.io.train_data
from flexneuart.config import TQDM_FILE
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.models.train.batching import TrainSamplerFixedChunkSize


def main_cli():
    parser = argparse.ArgumentParser('conversion from cedr to sentencebert format')

    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=str, nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=str, required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=str, required=True)

    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=str, required=True)
    
    parser.add_argument('--output_dir_name', metavar='Folder containing the training data in sentence bert format', help='SentenceBERT training data',
                        type=str, required=True)
    
    args = parser.parse_args()

    # Create the directory to store the sentence bert
    os.makedirs(args.output_dir_name, exist_ok=True)

    dataset = flexneuart.io.train_data.read_datafiles(args.datafiles)
    qrelf = args.qrels
    qrels = read_qrels_dict(qrelf)
    train_pairs_all = flexneuart.io.train_data.read_pairs_dict(args.train_pairs)
    valid_run = read_run_dict(args.valid_run)

    train_sampler = TrainSamplerFixedChunkSize(train_pairs=train_pairs_all,
                                               neg_qty_per_query=7,
                                               qrels=qrels,
                                               epoch_repeat_qty=1,
                                               do_shuffle=False)
    
    with open(args.output_dir_name + '/train.jsonl', "w") as train_file:
        for sample in train_sampler:
            d = {
            'qid': sample.qid,
            'pos_id': str(sample.pos_id),
            'pos_id_score' : float(sample.pos_id_score),
            'neg_ids' : [str(s) for s in sample.neg_ids],
            'neg_ids_score': [float(sc) for sc in sample.neg_id_scores]
            }
            train_file.write(json.dumps(d) + '\n')
        


if __name__ == '__main__':
    main_cli()