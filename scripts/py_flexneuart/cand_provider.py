"""
Access to FlexNeuART candidate providers (i.e., basic querying)
"""
from collections import namedtuple
from jnius import autoclass

from scripts.config import TEXT_FIELD_NAME, DOCID_FIELD
from scripts.py_flexneuart.utils import dict_to_hash_map

CandidateEntry = namedtuple('CandidateEntry', ['doc_id', 'score'])
JCandidateEntry = autoclass('edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry')
JCandidateProvider = autoclass('edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider')

PROVIDER_TYPE_LUCENE = JCandidateProvider.CAND_TYPE_LUCENE
PROVIDER_TYPE_NMSLIB = JCandidateProvider.CAND_TYPE_NMSLIB
PROVIDER_TYPE_TREC_RUNS = JCandidateProvider.CAND_TYPE_TREC_RUNS

PROVIDER_TYPE_LIST = [PROVIDER_TYPE_LUCENE, PROVIDER_TYPE_NMSLIB, PROVIDER_TYPE_TREC_RUNS]

FAKE_QUERY_ID='fake_query_id'

def create_cand_provider(resource_manager, provider_type, provider_uri, add_config_file=None):
    """Create a candidate provider (for basic querying).

    :param resource_manager:   a resource manager object
    :param provider_type:      a provider type
    :param provider_uri:       a provider index location (or address, e.g., for NMSLIB)
    :param add_config_file:    an optional provider configuration file (not needed for Lucene and NMSLIB)

    :return: a candidate provider object
    """
    if provider_type not in PROVIDER_TYPE_LIST:
        raise Exception(f'Unsupported provider type: {provider_type}, supported providers are: ' + ' '.join(PROVIDER_TYPE_LIST))

    # FlexNeuART is multi-thread and for each thread we may need a separate provider object
    # (if the provider is not thread-safe), but in Python we generate only one provider (as we
    # have no real threads anyways)
    return JCandidateProvider.createCandProviders(resource_manager,
                                       provider_type,
                                       provider_uri,
                                       add_config_file,
                                       1)[0]

def create_query_dict(query_text,
                    query_id=FAKE_QUERY_ID, field_name=TEXT_FIELD_NAME):
    """Create a Java HashMap instance with query information.

    :param query_text:       query text: *WHITE-SPACE* tokenized query tokens
    :param query_id:         a query ID (can be anything or just stick to default)
    :param field_name:       a field name (currently it's hardcoded in FlexNeuART anyways, so don't change this default)

    :return:
    """
    return dict_to_hash_map({DOCID_FIELD : str(query_id), field_name : query_text})


def run_query(cand_provider,
               top_qty,
               query_text,
               query_id=FAKE_QUERY_ID, field_name=TEXT_FIELD_NAME):
    """Run a query.

    :param cand_provider:    a candidate provider object
    :param top_qty:          a number of top-scored entries to return
    :param query_text:       query text: *WHITE-SPACE* tokenized query tokens
    :param query_id:         a query ID (can be anything or just stick to default)
    :param field_name:       a field name (currently it's hardcoded in FlexNeuART anyways, so don't change this default)

    :return: a tuple: # of entries found, an array of candidate entries: (document ID, score) objects
    """
    query = create_query_dict(query_text, query_id, field_name)
    cand_info = cand_provider.getCandidates(0, query, top_qty)

    return cand_info.mNumFound, \
           [CandidateEntry(doc_id=e.mDocId, score=e.mScore) for e in cand_info.mEntries]
