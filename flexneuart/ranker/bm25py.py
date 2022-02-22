#
#  Copyright 2014+ Carnegie Mellon University
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
#

"""
    A Python-level reimplementation of BM25 ranker. It produces very
    comparable (though slightly different) results from the Java bm25 ranker.

    You need to call configure_classpath() before using this functionality.
"""
from collections import Counter
from math import log

from flexneuart.text_proc import handle_case
from flexneuart.ranker.base import BaseRanker
from flexneuart.retrieval.fwd_index import get_forward_index

from flexneuart.retrieval.utils import DataEntryFields
from flexneuart.retrieval.cand_provider import CandidateEntry
from typing import List, Union, Tuple, Dict

class BM25Ranker(BaseRanker):
    """
        A Python version the BM25 ranker, which can be used in various experiments where
        document text is being modified on the fly without updating the document index.
    """
    def __init__(self, resource_manager,
                 query_field_name,
                 index_field_name,
                 idf_index_field_name,
                 text_proc_obj_query=None,
                 text_proc_obj_doc=None,
                 keep_case=False,
                 k1=1.2, b=0.75):
        """Reranker constructor.

        :param resource_manager:      a resource manager object
        :param index_field_name:      the name of the text field
        :param idf_index_field_name   the name of the field to extract IDF values
        :param keep_case:             do not lower case
        :param text_proc_obj_query:   a text processing object for the query that would typically
                                      lemmatize/stem text and optionally remove stop words
        :param text_proc_obj_doc:     a text processing object for the document
        :param query_field_name:      the name of the query field
        :param k1                     BM25 k1 parameter
        :param b                      BM25 b parameter

        """
        super().__init__()
        self.resource_manager = resource_manager
        self.k1 = k1
        self.b = b

        self.text_proc_obj_query = text_proc_obj_query
        self.text_proc_obj_doc = text_proc_obj_doc

        self.do_lower_case = not keep_case
        self.query_field_name = query_field_name

        self.fwd_indx = get_forward_index(resource_manager, index_field_name)
        self.fwd_parsed_index = get_forward_index(resource_manager, idf_index_field_name)

        assert self.fwd_parsed_index.indx.isParsed() or self.fwd_parsed_index.indx.isParsedText()

        self.doc_qty = self.fwd_parsed_index.get_doc_qty()
        self.inv_avg_doc_len = 1.0 / self.fwd_parsed_index.get_avg_doc_len()

    def calc_idf(self, word):
        word_entry = self.fwd_parsed_index.indx.getWordEntry(word)
        if word_entry is None:
            return 0
        n = word_entry.mWordFreq
        return log(1 + (self.doc_qty - n + 0.5) / (n + 0.5))

    def handle_case(self, text: str):
        return handle_case(self.do_lower_case, text)

    def score_candidates(self,
                         cand_list: List[Union[CandidateEntry, Tuple[str, float]]],
                         query_info_obj_or_dict: Union[DataEntryFields, dict]) -> Dict[str, float]:
        """Score, but does not rank, a candidate list obtained from the candidate provider.
           Note that this function may (though this is ranker-dependent) use all query field fields,
           not just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the candidate records
        :param query_info_obj:      a query information object

        :return:  a dictionary where keys are document IDs and values are document scores
        """
        query_text = self.get_query_text(query_info_obj_or_dict)

        if self.text_proc_obj_query is not None:
            query_text = self.text_proc_obj_query(query_text)

        query_text = self.handle_case(query_text)
        query_toks = query_text.split()
        query_terms_idfs = {w: self.calc_idf(w) for w in set(query_toks)}

        res = {}

        for doc_id, score in cand_list:
            doc_text = self.fwd_indx.get_doc_text(doc_id)
            if self.text_proc_obj_doc is not None:
                doc_text = self.text_proc_obj_doc(doc_text)
            doc_text = self.handle_case(doc_text)
            doc_toks = doc_text.split()
            doc_len = len(doc_toks)
            counts = Counter(doc_toks)
            score = 0
            for qterm in query_toks:
                tf = counts[qterm]
                if tf > 0:
                    qidf = query_terms_idfs[qterm]
                    norm_tf = (tf * (self.k1 + 1)) / \
                              (tf + self.k1 * (1 - self.b + self.b * doc_len * self.inv_avg_doc_len))
                    score += qidf * norm_tf

            res[doc_id] = score

        return res