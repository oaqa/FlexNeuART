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
    Convenience wrapper(c) for neural re-ranking models.
"""
import os
import torch

from flexneuart.config import DOCID_FIELD

from flexneuart.text_proc import handle_case
from flexneuart.models.base import ModelSerializer
from flexneuart.models.train.batch_obj import BatchObject
from flexneuart.models.train.amp import get_amp_processors
from flexneuart.models.train.batching import BatchingValidationGroupByQuery

from flexneuart.retrieval.utils import DataEntryFields
from flexneuart.retrieval.fwd_index import get_forward_index

from flexneuart.retrieval.cand_provider import CandidateEntry
from typing import List, Union, Tuple, Dict

from .base import BaseRanker


class NeuralRanker(BaseRanker):
    """
        A convenience wrapper(c) for neural re-ranking models.
        The constructor assumes that the model file path is relative to the collection (resource root) directory!
    """
    def __init__(self, resource_manager,
                       query_field_name,
                       index_field_name,
                       keep_case,
                       device_name, batch_size,
                       model_path_rel,
                       text_proc_obj_query=None,
                       text_proc_obj_doc=None,
                       cand_score_weight=0,
                       amp=False):
        """Reranker constructor.

        :param resource_manager:      a resource manager object
        :param query_field_name:      the name of the query field
        :param index_field_name:      the name of the text field
        :param keep_case:             do not lower case
        :param device_name:           a device name
        :param batch_size:            the size of the batch
        :param model_path_rel:        a path to a (previously trained) and serialized model relative to the resource root
        :param text_proc_obj_doc:     a text processing object for the document
        :param query_field_name:      the name of the query field
        :param cand_score_weight      a weight for the candidate generation scores
        :param amp                    if True, use automatic mixed precision

        """
        super().__init__()
        self.resource_manager = resource_manager
        # It is important to check before passing this to RankLib,
        # which does not handle missing files gracefully
        model_file_name_full_path = os.path.join(resource_manager.getResourceRootDir(), model_path_rel)
        if not os.path.exists(model_file_name_full_path):
            raise Exception(f'Missing model file: {model_file_name_full_path}')

        self.device_name = device_name
        self.cand_score_weight = cand_score_weight
        model_holder = ModelSerializer.load_all(model_file_name_full_path)

        self.model = model_holder.model
        self.model.to(device_name)

        self.amp = amp

        self.text_proc_obj_query = text_proc_obj_query
        self.text_proc_obj_doc = text_proc_obj_doc

        self.do_lower_case = not keep_case

        self.max_query_len = model_holder.max_query_len
        self.max_doc_len = model_holder.max_doc_len

        self.batch_size = batch_size

        self.query_field_name = query_field_name

        self.fwd_indx = get_forward_index(resource_manager, index_field_name)

    def handle_case(self, text: str):
        return handle_case(self.do_lower_case, text)

    def score_candidates(self, cand_list : List[Union[CandidateEntry, Tuple[str, float]]],
                               query_info_obj_or_dict : Union[DataEntryFields, dict]) -> Dict[str, float]:
        """Score, but does not rank, a candidate list obtained from the candidate provider.
           Note that this function may (though this is ranker-dependent) use all query field fields,
           not just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the candidate records
        :param query_info_obj:      a query information object

        :return:  a dictionary where keys are document IDs and values are document scores
        """
        query_info_obj_or_dict = self.get_query_info_obj(query_info_obj_or_dict)

        if type(query_info_obj_or_dict) == dict:
            query_text = query_info_obj_or_dict[self.query_field_name]
            query_id = query_info_obj_or_dict[DOCID_FIELD]
        else:
            if type(query_info_obj_or_dict) != DataEntryFields:
                raise Exception('A query object info type should be DataEntryFields or a dictionary!')
            query_text = query_info_obj_or_dict.getString(self.query_field_name)
            query_id = query_info_obj_or_dict.mEntryId

        query_text = self.handle_case(query_text)
        query_data = {query_id: query_text}

        doc_data = {}
        retr_score = {}

        for doc_id, score in cand_list:
            doc_text = self.fwd_indx.get_doc_text(doc_id)

            if self.text_proc_obj_doc is not None:
                doc_text = self.text_proc_obj_doc(doc_text)

            doc_text = self.handle_case(doc_text)
            doc_data[doc_id] = doc_text
            retr_score[doc_id] = score

        data_set = query_data, doc_data
        run = {query_id : retr_score}

        res = {}

        auto_cast_class, _ = get_amp_processors(self.amp)

        with torch.no_grad():
            self.model.eval()

            iter_val = BatchingValidationGroupByQuery(batch_size=self.batch_size,
                                                      dataset=data_set, model=self.model,
                                                      max_query_len=self.max_query_len,
                                                      max_doc_len=self.max_doc_len,
                                                      run=run)

            for batch in iter_val():
                with auto_cast_class():
                    batch: BatchObject = batch
                    batch.to(self.device_name)
                    model_scores = self.model(*batch.features)
                    assert len(model_scores) == len(batch)
                    scores = model_scores + batch.cand_scores * self.cand_score_weight
                    # tolist() works much faster compared to extracting scores one by one using .item()
                    scores = scores.tolist()

                    for qid, did, score in zip(batch.query_ids, batch.doc_ids, scores):
                        res[did] = score

        return res
