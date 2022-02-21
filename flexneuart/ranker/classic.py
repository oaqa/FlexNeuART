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
    Access to FlexNeuART rankers implemented in the Java layer.

    You need to call configure_classpath() before using this functionality.
"""
import os

from jnius import autoclass
from flexneuart.retrieval.cand_provider import JCandidateEntry, CandidateEntry
from .base import BaseRanker

from flexneuart.retrieval.utils import DataEntryFields
from typing import List, Union, Tuple, Dict

JDataPointWrapper = autoclass('edu.cmu.lti.oaqa.flexneuart.letor.DataPointWrapper')


class ClassicRanker(BaseRanker):
    """An interface to classic (non-neural) rankers, which are implemented a the Java layer.
       Model and configuration files are relative to the collection directory (resource root directory).
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

    def score_candidates(self, cand_list : List[Union[CandidateEntry, Tuple[str, float]]],
                               query_info_obj_or_dict : Union[DataEntryFields, dict]) -> Dict[str, float]:
        """Score, but does not rank, a candidate list obtained from the candidate provider.
           Note that this function may (though this is ranker-dependent) use all query field fields,
           not just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the candidate records
        :param query_info_obj:      a query information object

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

        for doc_id, _ in cand_list:
            feat = all_doc_feats.get(doc_id)
            assert feat is not None
            self.dp_wrapper.assign(feat)
            res[doc_id] = self.model.eval(self.dp_wrapper)

        return res

