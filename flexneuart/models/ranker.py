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
Access to FlexNeuART re-ranking functionality
"""
import os
import torch
from jnius import autoclass

from flexneuart.config import DOCID_FIELD

from flexneuart.models.base import ModelSerializer
from flexneuart.models.train.data import iter_valid_records
from flexneuart.models.train.data  import DOC_TOK_FIELD, DOC_MASK_FIELD, \
                                QUERY_TOK_FIELD, QUERY_MASK_FIELD,\
                                QUERY_ID_FIELD, DOC_ID_FIELD

from flexneuart.retrieval.utils import query_dict_to_dataentry_fields, DataEntryFields
from flexneuart.retrieval.cand_provider import JCandidateEntry
from flexneuart.retrieval.fwd_index import get_forward_index

JDataPointWrapper = autoclass('edu.cmu.lti.oaqa.flexneuart.letor.DataPointWrapper')

class BaseQueryRanker:
    """A base re-ranker class."""
    def get_query_info_obj(self, query_info_obj_or_dict):
        """an instance of
                i) a DataEntryFields object
                ii) a dictionary object, which will the function
                    try to convert to DataEntryFields

        :param query_info_obj_or_dict: a DataEntryFields object
        :return:
        """
        if type(query_info_obj_or_dict) == dict:
            query_info_obj = query_dict_to_dataentry_fields(query_info_obj_or_dict)
        else:
            if type(query_info_obj_or_dict) != DataEntryFields:
                raise Exception('A query object info type should be DataEntryFields or a dictionary!')
            query_info_obj = query_info_obj_or_dict
        return query_info_obj

    def rank_candidates(self, cand_list, query_info_obj_or_dict):
        """Score and rank a list of candidates obtained from the candidate provider.
           Note that this function may (though this is ranker-dependent) use all query field fields,
           not just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the objects of the type CandidateEntry
        :param query_info_obj:      an instance of
                                        i) a DataEntryFields object
                                        ii) a dictionary object, which will the function
                                            try to convert to DataEntryFields

        :return a list of tuples (document id, score) sorted in the order of decreasing scores
        """
        tmp_res = []

        for did, score in self.score_candidates(cand_list, query_info_obj_or_dict).items():
            tmp_res.append( (score, did))

        tmp_res.sort(reverse=True)

        return [ (did, score) for score, did in tmp_res]

    def score_candidates(self, cand_list, query_info_obj_or_dict):
        """Score, but does not rank, a candidate list obtained from the candidate provider.
           Note that this function may (though this is ranker-dependent) use all query field fields,
           not just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the objects of the type CandidateEntry
        :param query_info_obj:      an instance of
                                        i) a DataEntryFields object
                                        ii) a dictionary object, which will the function
                                            try to convert to DataEntryFields

        :return:  a dictionary where keys are document IDs and values are document scores
        """
        raise NotImplementedError



class JavaQueryRanker(BaseQueryRanker):
    """An interface to Java-layer re-rankers. Model and configuration files
        are relative to the collection directory (resource root directory).
    """
    def __init__(self, resource_manager, feat_extr_file_name, model_file_name):
        """Reranker constructor.

        :param resource_manager:      a resource manager object
        :param feat_extr_file_name:   feature extractor JSON configuration file.
        :param model_file_name:       a (previously trained/created) model file name
        """
        super().__init__()

        # It is important to check before passing this to RankLib,
        # which does not handle missing files gracefully
        model_file_name_full_path = os.path.join(resource_manager.getResourceRootDir(), model_file_name)
        if not os.path.exists(model_file_name_full_path):
            raise Exception(f'Missing model file: {model_file_name_full_path}')
        self.model = resource_manager.loadRankLibModel(model_file_name)
        self.feat_extr = resource_manager.getFeatureExtractor(feat_extr_file_name)
        self.dp_wrapper = JDataPointWrapper()

    def score_candidates(self, cand_list, query_info_obj_or_dict):
        """Rank a candidate list obtained from the candidate provider.
           Note that this function needs all relevant query fields, not
           just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the objects of the type CandidateEntry
        :param query_info_obj:      an instance of
                                        i) a DataEntryFields object
                                        ii) a dictionary object, which will the function
                                            try to convert to DataEntryFields

        :return:  a dictionary where keys are document IDs and values are document scores
        """
        query_info_obj = self.get_query_info_obj(query_info_obj_or_dict)

        res = {}
        # First convert a list of candidate back to the Java types
        # These operations have really small overhead compared to
        # actuall searching and re-ranking in most cases.
        cands_java = [JCandidateEntry(e.doc_id, e.score) for e in cand_list]
        # Compute a dictionary of features using Java api
        all_doc_feats = self.feat_extr.getFeatures(cands_java, query_info_obj)

        for cand in cand_list:
            feat = all_doc_feats.get(cand.doc_id)
            assert feat is not None
            self.dp_wrapper.assign(feat)
            res[cand.doc_id] = self.model.eval(self.dp_wrapper)

        return res


class PythonNNQueryRanker(BaseQueryRanker):
    "A neural Python-layer re-ranker. The model file name is relative to the collection (resource root) directory!"
    def __init__(self, resource_manager,
                       query_field_name,
                       index_field_name,
                       device_name, batch_size,
                       model_path_rel):
        """Reranker constructor.

        :param resource_manager:      a resource manager object
        :param query_field_name:      the name of the query field
        :param index_field_name:      the name of the text field
        :param device_name:           a device name
        :param batch_size:            the size of the batch
        :param model_path_rel:        a path to a (previously trained) and serialized model relative to the resource root

        """
        super().__init__()
        self.resource_manager = resource_manager
        # It is important to check before passing this to RankLib,
        # which does not handle missing files gracefully
        model_file_name_full_path = os.path.join(resource_manager.getResourceRootDir(), model_path_rel)
        if not os.path.exists(model_file_name_full_path):
            raise Exception(f'Missing model file: {model_file_name_full_path}')

        self.device_name = device_name
        model_holder = ModelSerializer.load_all(model_file_name_full_path)

        self.model = model_holder.model
        self.model.to(device_name)

        self.max_query_len = model_holder.max_query_len
        self.max_doc_len = model_holder.max_doc_len

        self.batch_size = batch_size

        self.query_field_name = query_field_name

        self.fwd_indx = get_forward_index(resource_manager, index_field_name)
        self.fwd_indx.check_is_text_raw()

    def score_candidates(self, cand_list, query_info_obj_or_dict):
        """Rank a candidate list obtained from the candidate provider.
           Note that this function needs all relevant query fields, not
           just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the objects of the type CandidateEntry
        :param query_info_obj:      an instance of
                                        i) a DataEntryFields object
                                        ii) a dictionary object, which will the function
                                            try to convert to DataEntryFields

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

        query_data = {query_id: query_text}

        doc_data = {}
        retr_score = {}
        run_did_and_scores = {}
        for doc_id, score in cand_list:
            doc_text = self.fwd_indx.get_doc_raw(doc_id)
            doc_data[doc_id] = doc_text
            retr_score[doc_id] = score
            run_did_and_scores[doc_id] = 0

        data_set = query_data, doc_data
        run = {query_id : run_did_and_scores}

        res = {}

        with torch.no_grad():
            for records in iter_valid_records(self.model, self.device_name, data_set, run,
                                              self.batch_size,
                                              self.max_query_len, self.max_doc_len):
                scores = self.model(records[QUERY_TOK_FIELD],
                                    records[QUERY_MASK_FIELD],
                                    records[DOC_TOK_FIELD],
                                    records[DOC_MASK_FIELD])

                scores = scores.tolist()

                for qid, did, score in zip(records[QUERY_ID_FIELD], records[DOC_ID_FIELD], scores):
                    res[did] = score

        return res


