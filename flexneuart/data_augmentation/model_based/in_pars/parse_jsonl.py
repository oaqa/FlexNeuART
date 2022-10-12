import json
import argparse
import csv
import os
import sys
import time
import numpy as np

def extract_docs(args):

    file = open(args.input_path)

    doc_file = open(os.path.join(args.output_path, "docs.tsv"), "w")
    query_file = open(os.path.join(args.output_path, "queries.tsv"), "w")

    doc_writer = csv.writer(doc_file, delimiter='\t')
    query_writer = csv.writer(query_file, delimiter='\t')

    time_stamp = time.time()
    query_id = "QP{0}_{1}_{2}"

    qrel_list = []
    counter = 0

    for json_line in file:
        line = json.loads(json_line)
        doc_writer.writerow(['doc', line['doc_id'], line['doc_text']])
        qid = query_id.format(time_stamp, line['doc_id'], counter)
        query_writer.writerow(['query', qid, line['question']])

        qrel_list.append([qid, line['doc_id'], np.mean(line['log_probs'])])
        counter += 1
    
    file.close()
    query_file.close()
    doc_file.close()

    score_sorted_qrels = sorted(qrel_list, key=lambda x: float(x[-1]), reverse=True)

    if args.topk is not None:
        num_keep = args.topk
    else:
        num_keep = int(len(qrel_list) * args.topf)
    
    score_sorted_qrels = score_sorted_qrels[:num_keep]

    qrel_file = open(os.path.join(args.output_path, "qrels.txt"), "w")
    qrel_writer = csv.writer(qrel_file, delimiter=' ')
    
    for row in score_sorted_qrels:
        qrel_writer.writerow([row[0], 0, row[1], 1])
    
    qrel_file.close()

    print("Data Extacted !!!!")
    
def perform_checks(input_path, output_path):
    if os.path.isfile(input_path)==False:
        print("Input is not a file or Path does not exist")
        sys.exit(1)
    
    if os.path.exists(output_path)==True:
        print("Output path exists, please sepcify another path")

    os.makedirs(output_path)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                    help="Path to the input jsonl file")
    parser.add_argument('--output_path', required=True,
                    help="path to keep the extracted output files")
    
    top_selection_group = parser.add_mutually_exclusive_group(required=True)
    top_selection_group.add_argument('--topk', type=int,
                                    help="selects the K queries with the highest log prob. score")
    top_selection_group.add_argument('--topf', type=float,
                                    help="selects the top f% of queries with highest log prob score")

    args = parser.parse_args()

    perform_checks(args.input_path, args.output_path)

    extract_docs(args)


    

