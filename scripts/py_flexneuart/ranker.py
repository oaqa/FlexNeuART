"""
Access to FlexNeuART re-ranking functionality
"""
import os
from jnius import autoclass

from scripts.py_flexneuart.cand_provider import create_query_dict, JCandidateEntry
from scripts.py_flexneuart.utils import dict_to_hash_map

JRankerFactory = autoclass('ciir.umass.edu.learning.RankerFactory')
JCompositeFeatureExtractor = autoclass('edu.cmu.lti.oaqa.flexneuart.letor.CompositeFeatureExtractor')
JDataPointWrapper = autoclass('edu.cmu.lti.oaqa.flexneuart.letor.DataPointWrapper')


class QueryRanker:
    def __init__(self, resource_manager, feat_extr_file_name, model_file_name):
        """Reranker constructor.

        :param resource_manager:   a resource manager object
        :param feat_extr_file_name:   feature extractor JSON configuration file.
        :param model_file_name:       a (previously trained/creaed) model file name
        """
        rank_factory = JRankerFactory()
        # It is important to check before passing this to RankLib,
        # which does not handle missing files gracefully
        if not os.path.exists(model_file_name):
            raise Exception(f'Missing model file: {model_file_name}')
        self.model = rank_factory.loadRankerFromFile(model_file_name)
        self.feat_extr = JCompositeFeatureExtractor(resource_manager, feat_extr_file_name)
        self.dp_wrapper = JDataPointWrapper()

    def rank_candidates(self, cand_list, query_info_dict):
        """Rank a candidate list obtained from the candidate provider.
           Note that this function needs all relevant query fields, not
           just a field that was used to retrieve the list of candidate entries!

        :param cand_list:           a list of the objects of the type CandidateEntry
        :param query_info_dict:     a dictionary containing all key query fields,
                                    which are necessary for re-ranking (can be both
                                    native Python dict too)

        :return:  a dictionary where keys are document IDs and values are document scores
        """
        if type(query_info_dict) == dict:
            query_info_dict = dict_to_hash_map(query_info_dict)

        res = {}
        # First convert a list of candidate back to the Java types
        # These operations have really small overhead compared to
        # actuall searching and re-ranking in most cases.
        cands_java = [JCandidateEntry(e.doc_id, e.score) for e in cand_list]
        # Compute a dictionary of features using Java api
        all_doc_feats = self.feat_extr.getFeatures(cands_java, query_info_dict)

        for cand in cand_list:
            feat = all_doc_feats.get(cand.doc_id)
            assert feat is not None
            self.dp_wrapper.assign(feat)
            res[cand.doc_id] = self.model.eval(self.dp_wrapper)

        return res


