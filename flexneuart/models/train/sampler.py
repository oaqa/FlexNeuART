#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR: https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
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
import numpy as np
from typing import List


class TrainSample:
    def __init__(self, qid :
                 str, pos_id : str, pos_id_score : float,
                 neg_ids : List[str], neg_id_scores : List[float]):
        """A simple (wrapper) object to hold sampling information.

        :param qid:             query ID
        :param pos_id:          positive document ID
        :param pos_id_score:    a query-document score for the positive document
        :param neg_ids:      an array of negative document IDs
        :param neg_id_scores:   query-document scores for negative documents
        """
        self.qid = qid
        self.pos_id = pos_id
        self.pos_id_score = pos_id_score
        self.neg_ids = neg_ids
        self.neg_id_scores = neg_id_scores

    def __str__(self):
        return f'query ID: {self.qid} pos. doc. ID/score: {self.pos_id}/{self.pos_id_score} ' + \
                   f'negative doc. IDs/scores: ' +  \
                   ', '.join([str(self.neg_ids[k]) + '/' + str(self.neg_id_scores[k]) for k in range(len(self.neg_id_scores))])


class TrainSamplerFixedChunkSize:

    """
        A helper class to sample training data in chunks of fixed size, which include
        one positive and multiple negative document IDs.
    """
    def __init__(self,
                     train_pairs:dict,
                     neg_qty_per_query:int,
                     qrels:dict,
                     epoch_repeat_qty:int = 1,
                     do_shuffle: bool = True
                 ):
        """Constructor.

        :param train_pairs:         a dictionary of query document pairs, where keys are query IDs,
                                    and values are document IDs for which we have document texts.
        :param neg_qty_per_query:   a number of negative samples per query
        :param qrels:               q QREL (relevance information) dictionary: if a query
                                    has fewer documents available, it is ignored.
        :param epoch_repeat_qty:    a number of times we should repeat/replicate each epoch
        :param do_shuffle:          true to shuffle training queries
        """
        self.neg_qty_per_query = neg_qty_per_query

        self.qids = list(train_pairs.keys())
        self.query_qty = len(self.qids)
        self.step_qty = epoch_repeat_qty * self.query_qty
        self.step = 0
        self.qrels = qrels
        self.train_pairs = train_pairs
        self.do_shuffle = do_shuffle
        if self.do_shuffle:
            np.random.shuffle(self.qids)

        self.qnum = -1

    def get_chunk_size(self):
        """Return the number of entries in every chunk."""
        return 1 + self.neg_qty_per_query

    def __iter__(self):
        self.qnum = -1
        self.step = 0
        return self

    def __next__(self) -> TrainSample:

        while self.step < self.step_qty:
            self.qnum += 1
            if self.qnum >= self.query_qty:
                self.qnum = 0
                if self.do_shuffle:
                    np.random.shuffle(self.qids)

            self.step += 1
            qid = self.qids[self.qnum]
            query_train_pairs = self.train_pairs[qid]

            pos_ids = [did for did in query_train_pairs if self.qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) < 1:
                continue
            pos_id = np.random.choice(pos_ids, size=1)[0]
            pos_ids_lookup = set(pos_ids)

            all_neg_id_arr = [did for did in query_train_pairs if did not in pos_ids_lookup]
            all_neg_qty = len(all_neg_id_arr)
            if all_neg_qty < 1:
                continue
            if all_neg_qty >= self.neg_qty_per_query:
                neg_id_arr = list(np.random.choice(all_neg_id_arr,
                                                  size=self.neg_qty_per_query,
                                                  replace=False))
            else:
                assert self.neg_qty_per_query - all_neg_qty > 0
                # If we don't have enough negatives, it's best to use all of them
                # deterministically and a few missing randomly. If we just sample
                # with replacement, there will be unnecessarily missing items.
                neg_id_arr = all_neg_id_arr + list(np.random.choice(all_neg_id_arr,
                                                  size=self.neg_qty_per_query - all_neg_qty,
                                                  replace=True))

            return TrainSample(qid=qid,
                               pos_id=pos_id, pos_id_score = query_train_pairs[pos_id],
                               neg_ids=neg_id_arr,
                               neg_id_scores=[query_train_pairs[neg_id] for neg_id in neg_id_arr])

        raise StopIteration
