#!/usr/bin/env python
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
import torch

from flexneuart.models.train.sampler import TrainSamplerFixedChunkSize, TrainSample
from flexneuart.models.base import BaseModel
from flexneuart.models.train.batch_obj import BatchObject

PAD_CODE=1 # your typical padding symbol


class TempBatch:
    def __init__(self):
        self.query_ids = []
        self.query_texts = []
        self.doc_ids = []
        self.doc_texts = []
        self.labels = []
        self.cand_scores = []


class BatchingBase:
    """The base batching class, which provides training and validation batches."""
    def __init__(self, batch_size, dataset,
                        model,
                        max_query_len, max_doc_len):
        self.batch_size = batch_size
        self.dataset = dataset
        self.model = model
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        self.batch = []

    def _add_to_batch(self, qid, query_text, did, doc_text, cand_score, label):
        batch: TempBatch = self.batch

        batch.query_ids.append(qid)
        batch.query_texts.append(query_text)
        batch.doc_ids.append(did)
        batch.doc_texts.append(doc_text)
        batch.cand_scores.append(cand_score)
        batch.labels.append(label)

    def _batchify(self) -> BatchObject:
        """Batchify data and clear the batch content."""
        model: BaseModel = self.model
        batch: TempBatch = self.batch

        features = model.featurize(max_query_len=self.max_query_len,
                                   max_doc_len=self.max_doc_len,
                                   query_texts=batch.query_texts,
                                   doc_texts=batch.doc_texts)

        res = BatchObject(query_ids=batch.query_ids,
                          doc_ids=batch.doc_ids,
                          labels=torch.LongTensor(batch.labels),
                          cand_scores=torch.FloatTensor(batch.cand_scores),
                          features=features)

        self.batch = TempBatch()

        return res


class BatchingValidationGroupByQuery(BatchingBase):
    """
        A validation batching class, which has an important property
        that it never "crosses" query boundaries.
    """
    def __init__(self,  batch_size : int,
                        dataset : tuple,
                        model : BaseModel,
                        max_query_len, max_doc_len,
                        run : dict
                 ):
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         model=model,
                         max_query_len=max_query_len, max_doc_len=max_doc_len)
        self.run = run

    def __call__(self):
        ds_queries, ds_docs = self.dataset
        self.batch = TempBatch()

        for qid in self.run:
            query_text = ds_queries.get(qid)
            assert query_text is not None, f'Missing query text: {qid}'
            for did, score in self.run[qid].items():
                doc_text = ds_docs.get(did)
                assert doc_text is not None, f'Missing document ID: {did}'

                self._add_to_batch(qid=qid, query_text=query_text,
                                   did=did, doc_text=doc_text,
                                   cand_score=score, label=0)

                if len(self.batch.query_ids) >= self.batch_size:
                    # clears bach too!
                    yield self._batchify()

            # because some models cannot generate features for mixed-query batches
            # we have to clear the batch here
            if len(self.batch.query_ids) > 0:
                # clears bach too!
                yield self._batchify()


class BatchingTrainFixedChunkSize(BatchingBase):
    """
        A standard train-time batching class, which always produces tensors whose size
        are a multiple of the negative # of samples + 1.
    """
    def __init__(self,  batch_size : int,
                        dataset : tuple,
                        model : BaseModel,
                        max_query_len, max_doc_len,
                        train_sampler: TrainSamplerFixedChunkSize,
                 ):
        super().__init__(batch_size=batch_size,
                         dataset=dataset,
                         model=model,
                         max_query_len=max_query_len, max_doc_len=max_doc_len)
        self.train_sampler = train_sampler

    def __call__(self):

        ds_queries, ds_docs = self.dataset
        self.batch = TempBatch()

        for qobj in self.train_sampler:
            qobj : TrainSample = qobj

            query_text = ds_queries.get(qobj.qid)
            assert query_text is not None, f'Missing query ID: {qobj.qid}'

            neg_qty = len(qobj.neg_ids)
            assert(neg_qty) == len(qobj.neg_id_scores)
            assert(neg_qty) == self.train_sampler.neg_qty_per_query

            pos_doc_text = ds_docs.get(qobj.pos_id)
            assert pos_doc_text is not None, f'Missing document ID: {qobj.pos_id}'

            self._add_to_batch(qid=qobj.qid, query_text=query_text,
                               did=qobj.pos_id, doc_text=pos_doc_text,
                               cand_score=qobj.pos_id_score, label=1)

            for neg_id, neg_score in zip(qobj.neg_ids, qobj.neg_id_scores):
                neg_doc_text=ds_docs.get(neg_id)
                assert neg_doc_text is not None, f'Missing document ID: {neg_id}'
                self._add_to_batch(qid=qobj.qid, query_text=query_text,
                                   did=neg_id, doc_text=neg_doc_text,
                                   cand_score=neg_score, label=0)

            batch: TempBatch = self.batch

            if len(batch.query_ids) // self.train_sampler.get_chunk_size() >= self.batch_size:
                # clears bach too!
                yield self._batchify()

