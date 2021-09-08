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
Access to FlexNeuART candidate providers (i.e., basic querying)
"""

from collections import namedtuple
from jnius import autoclass

from flexneuart.config import TEXT_FIELD_NAME
from flexneuart.retrieval.utils import query_dict_to_dataentry_fields, DataEntryFields

CandidateEntry = namedtuple('CandidateEntry', ['doc_id', 'score'])
JCandidateEntry = autoclass('edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry')
JCandidateProvider = autoclass('edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateProvider')

PROVIDER_TYPE_LUCENE = JCandidateProvider.CAND_TYPE_LUCENE
PROVIDER_TYPE_NMSLIB = JCandidateProvider.CAND_TYPE_NMSLIB
PROVIDER_TYPE_TREC_RUNS = JCandidateProvider.CAND_TYPE_TREC_RUNS

PROVIDER_TYPE_LIST = [PROVIDER_TYPE_LUCENE, PROVIDER_TYPE_NMSLIB, PROVIDER_TYPE_TREC_RUNS]

FAKE_QUERY_ID='fake_query_id'

def create_cand_provider(resource_manager, provider_type, provider_uri, add_config_file=None):
    """Create a candidate provider (for basic querying). Configuration and
       index file paths are relative to the collection root (stored in the resource manager)

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
    return resource_manager.createCandProviders(provider_type,
                                                   provider_uri,
                                                   add_config_file,
                                                   1)[0]


def create_text_query_obj(query_text,
                          query_id=FAKE_QUERY_ID, field_name=TEXT_FIELD_NAME):
    """Create a Java object with text query information.

    :param query_text:       query text: *WHITE-SPACE* tokenized query tokens
    :param query_id:         a query ID (can be anything or just stick to default)
    :param field_name:       a field name (currently it's hardcoded in FlexNeuART anyways, so don't change this default)

    :return:
    """
    obj = DataEntryFields(str(query_id))
    obj.setString(field_name, query_text)
    return obj


def run_query_internal(cand_provider, top_qty, query_obj):
    """An auxilliary function not intended to be used directly"""
    cand_info = cand_provider.getCandidates(0, query_obj, top_qty)

    return cand_info.mNumFound, \
           [CandidateEntry(doc_id=e.mDocId, score=e.mScore) for e in cand_info.mEntries]


def run_text_query(cand_provider,
               top_qty,
               query_text,
               query_id=FAKE_QUERY_ID, field_name=TEXT_FIELD_NAME):
    """Run a single-field text query.

    :param cand_provider:    a candidate provider object
    :param top_qty:          a number of top-scored entries to return
    :param query_text:       query text: *WHITE-SPACE* tokenized query tokens
    :param query_id:         a query ID (can be anything or just stick to default)
    :param field_name:       a field name (currently it's hardcoded in FlexNeuART anyways, so don't change this default)

    :return: a tuple: # of entries found, an array of candidate entries: (document ID, score) objects
    """
    query_obj = create_text_query_obj(query_text, query_id, field_name)

    return run_query_internal(cand_provider, top_qty, query_obj)


def run_query(cand_provider,
            top_qty,
            query_dict,
            default_query_id=FAKE_QUERY_ID):
    """Run a generic (not-necessarily single-field text) query.

    :param cand_provider:       a candidate provider object
    :param top_qty:             a number of top-scored entries to return
    :param query_dict:          query key-value dictionary that may or may not have the query/doc ID
    :param default_query_id:    a default query ID to use if query_dict has none.

    :return: a tuple: # of entries found, an array of candidate entries: (document ID, score) objects
    """
    if type(query_dict) != dict:
        raise Exception('A query object should be a dictionary!')
    query_obj = query_dict_to_dataentry_fields(query_dict, default_query_id)

    return run_query_internal(cand_provider, top_qty, query_obj)
