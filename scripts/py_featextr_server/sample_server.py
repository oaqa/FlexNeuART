#!/usr/bin/env python
# This is a sample server that illustrates how to generate query-document scores
# and pass them to the client.
import sys

sys.path.append('.')

from scripts.py_featextr_server.base_server import BaseQueryHandler, start_query_server, \
    SAMPLE_HOST, SAMPLE_PORT


# Exclusive==True means that only one get_scores
# function is executed at at time
class SampleQueryHandler(BaseQueryHandler):
    def __init__(self, exclusive=True):
        super().__init__(exclusive)

    # This function needs to be overridden
    def compute_scores_from_parsed_override(self, query, docs):
        print('get_scores', query.id, self.text_entry_to_str(query))
        sample_ret = {}
        for e in docs:
            print(self.text_entry_to_str(e))
            # Note that each element must be an array, b/c
            # we can generate more than one feature per document!
            sample_ret[e.id] = [0]
        return sample_ret

    # This function needs to be overridden
    def compute_scores_from_raw_override(self, query, docs):
        print('get_scores', query.id, query.text)
        sample_ret = {}
        for e in docs:
            print(e.text)
            # Note that each element must be an array, b/c
            # we can generate more than one feature per document!
            sample_ret[e.id] = [0]
        return sample_ret


if __name__ == '__main__':
    multi_threaded = True
    start_query_server(SAMPLE_HOST, SAMPLE_PORT, multi_threaded, SampleQueryHandler(exclusive=False))
