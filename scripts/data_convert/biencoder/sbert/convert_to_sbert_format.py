#!/usr/bin/env python
import argparse
import json

import flexneuart.io.train_data
from flexneuart.io.json import read_json, save_json
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.io.runs import read_run_dict

from batching import TrainSamplerFixedChunkSizeUnique

from tqdm import tqdm

def main_cli():
    parser = argparse.ArgumentParser('Convert CEDR format to sentence bert triplets')
    
    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=str, nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=str, required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=str, required=True)
    
    parser.add_argument('--output_dir', metavar='path to store the file containing training triplets', help='output path',
                        type=str, required=True)
    
    parser.add_argument('--neg_qty_per_query', metavar='listwise negatives',
                        help='Number of negatives per query for a listwise loss',
                        type=int, default=2)

    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=str, required=True)

    args = parser.parse_args()

    queries, docs = flexneuart.io.train_data.read_datafiles(args.datafiles)

    qrelf = args.qrels
    qrels = read_qrels_dict(qrelf)

    train_pairs_all = flexneuart.io.train_data.read_pairs_dict(args.train_pairs)

    valid_run = read_run_dict(args.valid_run)
    
    expected_num_train_pairs = 0
    for query_id in train_pairs_all.keys():
        expected_num_train_pairs += len(train_pairs_all[query_id].keys())

    train_sampler = TrainSamplerFixedChunkSizeUnique(train_pairs=train_pairs_all,
                                               neg_qty_per_query=args.neg_qty_per_query,
                                               qrels=qrels,
                                               epoch_repeat_qty=1,
                                               do_shuffle=False)

    iterator = iter(train_sampler)

    train_triplets = []
    while True:
        try:
            group = next(iterator)

            # for each query club the positive doc id with a negative doc id
            qid = group.qid
            pos_id = group.pos_id
            neg_ids = group.neg_ids

            for neg_id in neg_ids:
                train_triplets.append((qid, pos_id, neg_id))
        except StopIteration:
            break

    triplet_id_file_path = args.output_dir + "/train_triplets_id.jsonl"
    with open(triplet_id_file_path, 'w') as triplet_ids_file:
        for qid, pos_id, neg_id in train_triplets:
            d = {'qid': qid, 'positive_docid': pos_id, 'negative_doc_id': neg_id}
            triplet_ids_file.write(json.dumps(d) + '\n')
    
    triplet_file_path = args.output_dir + "/train_triplets.jsonl"
    with open(triplet_file_path, 'w') as triplet_file:
        for qid, pos_id, neg_id in train_triplets:
            d = {'anchor': queries[qid], 'positive': docs[pos_id], 'negative': docs[neg_id]}
            triplet_file.write(json.dumps(d) + '\n')

    eval_file_path = args.output_dir + "/eval.json"
    with open(eval_file_path, 'w') as eval_file:
        pos_query_qty = 0
        for qid, query_dict in valid_run.items():
            assert qid in queries, f'Missing query text {qid}, validation run'
            positive = []
            negative = []
            has_pos = 0
            for did, _ in query_dict.items():
                if qrels[qid].get(did, 0) > 0:
                    has_pos = 1
                    positive.append(docs[did])
                else:
                    negative.append(docs[did])
            pos_query_qty += has_pos

            d = {'query': queries[qid], 'positive': positive, 'negative': negative}
            eval_file.write(json.dumps(d) + '\n')

    print('Successfully wrote triplet ids file at ', triplet_id_file_path)
    print('Successfully wrote triplet file at ', triplet_file_path)
    print(f'Successfully wrote validation triplets at {eval_file_path}, # of queries: {len(valid_run)}, # of queries with at least one relevant: {pos_query_qty}')
if __name__ == '__main__':
    main_cli()