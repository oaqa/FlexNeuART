import random
import os
import csv
import sys
import time
import argparse
from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.fwd_index import get_forward_index

configure_classpath()
from flexneuart.retrieval.cand_provider import *

class BM25Retriever():
    def __init__(self, args):
        self.args = args
        
        collection, collection_root = self.__sanity_check()

        self.resource_manager = create_featextr_resource_manager(
            resource_root_dir=f'{collection_root}/{collection}/',
            fwd_index_dir='forward_index', model1_root_dir=f'derived_data/giza',
            embed_root_dir=f'derived_data/embeddings')

        self.cand_prov = create_cand_provider(self.resource_manager, PROVIDER_TYPE_LUCENE, args.index,
                                                add_config_file=args.bm25_config)
        self.counter = 0
        self.timestamp = time.time()

    def __call__(self, query):
        if len(query) == 0:
            return None, None, None
        query_response = run_text_query(self.cand_prov, self.args.max_docs_retrieved, query)
        no_of_responses =  query_response[0]

        if no_of_responses == 0:
            return None, None, None

        random_number = random.randint(0, min(no_of_responses, self.args.max_docs_retrieved)-1)

        negative_doc = query_response[1][random_number]
        negative_doc_id = negative_doc.doc_id

        raw_index = get_forward_index(self.resource_manager, 'text_raw')
        negative_doc_text = raw_index.get_doc_text_raw(negative_doc_id)

        query_id = self.__generate_unique_qid()

        return query_id, negative_doc_id, negative_doc_text


    def __generate_unique_qid(self):
        qid = str(self.timestamp)  + '_' + str(self.counter)
        self.counter += 1
        return qid 

    def __sanity_check(self):

        collection = os.getenv("COLLECTION")
        collection_root = os.getenv("COLLECTION_ROOT")

        if collection=='' or collection==None:
            print("Please export COLLECTION. Example -> export COLLECTION=msmarco_pass")
            sys.exit(1)     
        
        if collection_root=='' or collection_root==None:
            print("Please export COLLECTION_ROOT. Example -> export COLLECTION_ROOT=/home/ubuntu/efs/capstone/data")
            sys.exit(1)
        
        return collection, collection_root


def generate_negative(args):
    
    new_queries_file = open(args.aug_query)
    output_data = open(args.neg_doc, 'w')
    output_qrels = open(args.neg_doc_qrels, 'w')

    tsv_writer = csv.writer(output_data, delimiter='\t')
    tsv_writer_qrels = csv.writer(output_qrels, delimiter=' ')

    bm25_retriever = BM25Retriever(args)

    for line in new_queries_file:
        splits = line[:-1].split("\t", 2)
        query_id = splits[0]
        query = splits[2].rsplit("\t", 1)[0]

        _, negative_doc_id, negative_doc_text = bm25_retriever(query)
        if negative_doc_text==None:
            continue
        query_row = ['query', query_id, query]
        tsv_writer.writerow(query_row)

        doc_row_neg = ['doc', negative_doc_id, negative_doc_text]
        tsv_writer.writerow(doc_row_neg)

        tsv_writer_qrels.writerow([query_id, 0, negative_doc_id, -1])
    
    new_queries_file.close()
    output_data.close()
    output_qrels.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.aug_query = '/home/ubuntu/efs/capstone/data_aug/FlexNeuART/flexneuart/data_augmentation/generate_negative_docs/neg_qfile.tsv'
    args.neg_doc = './output/neg_doc.tsv'
    args.neg_doc_qrels = './output/neg_doc_qrels.txt'
    args.index = 'lucene_index'
    args.max_docs_retrieved = 100
    generate_negative(args)
