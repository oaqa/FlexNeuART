import os
import json


def generate_files(args):

    # ensure directory exists, file does not exist
    qrels_final_file = open(os.path.join(args.output_directory, 'qrels.txt'), 'w')
    train_pairs_final_file = open(os.path.join(args.output_directory, 'train_pairs.tsv'), 'w')

    for file_name, weight in zip([args.original_qrels, args.aug_query_qrels, args.neg_doc_qrels],
                                 [args.original_weight, args.aug_query_weight, args.neg_doc_weight]):
        f = open(file_name, 'r')
        for line in f.readlines():
            qrels_final_file.write(line + '\n')
            data = line.split()
            train_pairs_final_file.write('\t'.join(data[0], data[2], weights) + '\n')

    qrels_final_file.close()
    train_pairs_final_file.close()

    data_final_file = open(os.path.join(args.output_directory, 'data.tsv'), 'w')

    for file_name in [args.original_query, args.original_doc, args.aug_query, args.neg_doc]:
        f = open(file_name, 'r')
        for line in f.readlines():
            data_final_file.write(line + '\n')

    data_final_file.close()
