#!/usr/bin/env python
import sys
import argparse

sys.path.append('.')

from scripts.py_featextr_server.base_server import BaseQueryHandler, start_query_server

import numpy as np

from scripts.py_featextr_server.utils import load_embeddings, create_embed_map, robust_cosine_simil

# Exclusive==True means that only one get_scores
# function is executed at at time
class CosineSimilQueryHandler(BaseQueryHandler):
    def __init__(self, query_embed_file, doc_embed_file, exclusive, debug_print=False, use_idf=True):
        super().__init__(exclusive)

        self.debug_print = debug_print
        self.use_idf = use_idf

        print('Loading answer embeddings from: ' + doc_embed_file)
        answ_words, self.answ_embed = load_embeddings(doc_embed_file)
        self.answ_embed_map = create_embed_map(answ_words)

        if query_embed_file is not None:
            print('Loading query embeddings from: ' + query_embed_file)
            query_words, self.query_embed = load_embeddings(query_embed_file)
            self.query_embed_map = create_embed_map(query_words)
        else:
            self.query_embed = self.answ_embed
            self.query_embed_map = self.answ_embed_map
        print('Loading is done!')

    def text_entry_to_str(self, te):
        arr = []
        if self.debug_print:
            for winfo in te.entries:
                arr.append('%s %g %d ' % (winfo.word, winfo.IDF, winfo.qty))
        return 'doc_id=' + te.id + ' ' + ' '.join(arr)

    def create_doc_embed(self, is_query, text_entry):

        if is_query:
            embeds = self.query_embed
            embed_map = self.query_embed_map
        else:
            embeds = self.answ_embed
            embed_map = self.answ_embed_map

        zerov = np.zeros_like(embeds[0])
        res = zerov

        for winfo in text_entry.entries:
            vect_mult = winfo.qty
            if self.use_idf:
                vect_mult *= winfo.IDF
            word = winfo.word
            if word in embed_map:
                res += embeds[embed_map[word]] * vect_mult

        return res

    # This function overrides the parent class
    def compute_scores_from_parsed_override(self, query, docs):
        if self.debug_print:
            print('get_scores', query.id, self.text_entry_to_str(query))
        ret = {}
        query_embed = self.create_doc_embed(True, query)
        if self.debug_print:
            print(query_embed)
        for d in docs:
            if self.debug_print:
                print(self.text_entry_to_str(d))
            doc_embed = self.create_doc_embed(False, d)
            if self.debug_print:
                print(doc_embed)
            # Regular cosine deals poorly with all-zero vectors
            simil = robust_cosine_simil(doc_embed, query_embed)
            # simil = (1-cosine(doc_embed, query_embed))

            # Note that each element must be an array, b/c
            # we can generate more than one feature per document!
            ret[d.id] = [simil]

        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving word-embedding models.')

    parser.add_argument('--query_embed', metavar='query embeddings',
                        default=None, type=str,
                        help='Optional query embeddings file')

    parser.add_argument('--doc_embed', metavar='doc embeddings',
                        required=True, type=str,
                        help='document embeddings file')

    parser.add_argument('--debug_print', action='store_true',
                        help='Provide debug output')

    parser.add_argument('--port', metavar='server port',
                        required=True, type=int,
                        help='Server port')

    parser.add_argument('--host', metavar='server host',
                        default='127.0.0.1', type=str,
                        help='server host addr to bind the port')

    args = parser.parse_args()

    multi_threaded = True
    start_query_server(args.host, args.port, multi_threaded,
                     CosineSimilQueryHandler(exclusive=False,
                                             query_embed_file=args.query_embed,
                                             doc_embed_file=args.doc_embed,
                                             debug_print=args.debug_print))
