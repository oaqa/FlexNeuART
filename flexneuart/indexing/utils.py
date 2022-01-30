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
Access to FlexNeuART indexers
"""
import os
import shutil

from jnius import autoclass

from flexneuart.config import ANSWER_FILE_JSONL_GZ, MAX_INT32, \
                            ANSWER_FILE_JSON, ANSWER_FILE_BIN, QUESTION_FILE_JSON, QUESTION_FILE_BIN

UNKNOWN_TYPE='unknown'
# There seems to be no/less harm in "over-specify" the number of expected entries
# as opposed to under-specifying, which slows indexing down quite a bit
EXPECTED_INDEX_QTY=int(1e9)

#
# This is a somewhat unfinished version. It can be used, but it lacks data-file
# discovery functionality. It also uses absolute paths rather than paths
# relative to the collection root.
#

JLuceneIndexer = autoclass('edu.cmu.lti.oaqa.flexneuart.apps.LuceneIndexer')
JBuildFwdIndexApp = autoclass('edu.cmu.lti.oaqa.flexneuart.apps.BuildFwdIndexApp')
JForwardIndex=autoclass('edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex')

# These definitions are somewhat hard to extract from Java API (this seems
# to require writing functions returning these values), so it is easier to
# just hardcode it here, but these values must be in sync with Java code.
# see definitions inside edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex
# We have some checks if this is true in the Python code though (see below).

# Storage backend type
STORE_TYPE_MAPDB = 'mapdb'
STORE_TYPE_LUCENE = 'lucene'

# Checking if we are in sync with Java API
STORE_TYPES = [STORE_TYPE_MAPDB, STORE_TYPE_LUCENE]
assert set(STORE_TYPES) == set(JForwardIndex.getStoreTypeList().split(',')), \
       'Java and Python API are out of sync in terms of constants!'

# Storage types, a data dictionary is a key-value store
# where data values are stored as dictionary values.
# However, the offsetDict stores data in the form of a plain
# file and the key-value store keeps offsets.
FORWARD_INDEX_TYPE_DATA_DICT = 'dataDict'
FORWARD_INDEX_TYPE_OFFSET_DICT = 'offsetDict'
# In-memory index used for testing purposes only
FORWARD_INDEX_TYPE_INMEM = 'inmem'

FORWARD_INDEX_TYPES = [FORWARD_INDEX_TYPE_DATA_DICT, FORWARD_INDEX_TYPE_OFFSET_DICT, FORWARD_INDEX_TYPE_INMEM]
assert set(FORWARD_INDEX_TYPES) == set(JForwardIndex.getIndexTypeList().split(',')), \
       'Java and Python API are out of sync in terms of constants!'

# Various field types
# Unparsed, i.e., raw text
FORWARD_INDEX_FIELD_TYPE_TEXT_RAW = 'textRaw'
# Binary data
FORWARD_INDEX_FIELD_TYPE_BINARY = 'binary'
# Parsed text with positional information
FORWARD_INDEX_FIELD_TYPE_PARSED_TEXT = 'parsedText'
# Parsed text, bag-of-words (BOW) data, i.e., no positional information
FORWARD_INDEX_FIELD_TYPE_PARSED_TEXT_BOW = 'parsedBOW'

FORWARD_INDEX_FIELD_TYPES = [FORWARD_INDEX_FIELD_TYPE_TEXT_RAW, FORWARD_INDEX_FIELD_TYPE_BINARY,
                             FORWARD_INDEX_FIELD_TYPE_PARSED_TEXT, FORWARD_INDEX_FIELD_TYPE_PARSED_TEXT_BOW]
assert set(FORWARD_INDEX_FIELD_TYPES) == set(JForwardIndex.getIndexFieldTypeList().split(',')), \
       'Java and Python API are out of sync in terms of constants!'


class InputDataInfo:
    def __init__(self, index_subdirs, query_subdirs, data_file_name):
        self.index_subdirs = index_subdirs
        self.query_subdirs = query_subdirs
        self.data_file_name = data_file_name


def get_index_query_data_dirs(top_dir : str) -> InputDataInfo:
    """
       This is a 'discovery' function, which
            (1) finds all data and query files in a given top-level directory,
            (2) checks basic consistency.

       For example, if some query/question files have binary data file and some do not, it will raise an exception.

       Note that the files are supposed to be found in immediate sub-directories, i.e., there is no recursive search.

        :param top_dir: a top-level directory whose sub-directories contain data and/or query-files.
        :return collection-input-data-information object
    """
    assert os.path.isdir(top_dir), f'Not a directory: {top_dir}'
    subdir_list = [subdir for subdir in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, subdir))]

    index_subdirs = []
    query_subdirs = []
    data_file_name = None

    for subdir in subdir_list:
        print(f'Checking sub-directory: {subdir}')

        has_data = False
        for suff in ["", ".gz", ".bz2"]:
            data_fn = ANSWER_FILE_JSON + suff
            fn = os.path.join(top_dir, subdir, data_fn)
            if os.path.exists(fn) and os.path.isfile(fn):
                print(f'Found indexable data file: {fn}')
                has_data = True
                if data_file_name is None:
                    data_file_name = data_fn
                elif data_file_name != data_fn:
                    raise Exception(f'Inconsistent compression type for data files: {data_file_name} and {data_fn}')
        if has_data:
            index_subdirs.append(subdir)
        else:
            fn_bin = os.path.join(subdir, ANSWER_FILE_BIN)
            if os.path.exists(fn_bin):
                raise Exception(f'Inconsistent data setup in sub-directory {subdir}: ' +
                                f'the binary data file is present, but not an (un)compressed JSONL file')

        fn = os.path.join(top_dir, subdir, QUESTION_FILE_JSON)
        fn_bin = os.path.join(top_dir, subdir, QUESTION_FILE_BIN)
        if os.path.exists(fn):
            query_subdirs.append(subdir)
        else:
            if os.path.exists(fn_bin):
                raise Exception(f'Inconsistent query setup in sub-directory {subdir}:' +
                                 'the binary query file is present, but not an (un)compressed JSONL file')

    return InputDataInfo(index_subdirs=index_subdirs, query_subdirs=query_subdirs, data_file_name=data_file_name)


def create_lucene_index(input_data_dir,
                        output_dir_name,
                        index_field_name,
                        exact_match=False,
                        max_num_rec=MAX_INT32):
    """Call a Java function to create a Lucene index. This function first uses a "data-discovery"
       function get_index_query_data_dirs to find out all data files located in sub-directories
       of a given top-level directory.

        :param input_data_dir: a root folder containing processed input data.
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param exact_match: if true, we create a regular string index, which can be used
                            to retrieve documents using secondary IDs.
        :param max_num_rec: an optional number for the maximum number of records to index.
    """
    data_info: InputDataInfo = get_index_query_data_dirs(input_data_dir)
    if not data_info.index_subdirs or data_info.data_file_name is None:
        raise Exception(f'No indexable data is found in {input_data_dir}')
    create_lucene_index_explicit_subdirs(input_data_dir=input_data_dir,
                                          subdir_list=data_info.index_subdirs,
                                          output_dir_name=output_dir_name,
                                          index_field_name=index_field_name,
                                          exact_match=exact_match,
                                          datafile_name=data_info.data_file_name,
                                          max_num_rec=max_num_rec)


def create_forward_index(input_data_dir,
                        output_dir_name,
                        index_field_name,
                        field_type,
                        index_type,
                        store_type=STORE_TYPE_MAPDB,
                        max_num_rec=MAX_INT32,
                        expected_qty=EXPECTED_INDEX_QTY):
    """Call a Java function to create a forward index. This function first uses a "data-discovery"
       function get_index_query_data_dirs to find out all data files located in sub-directories
       of a given top-level directory.

        :param input_data_dir: a root folder containing processed input data.
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param index_type: a type of the index: inmem, dataDict, offsetDict
        :param field_type: a field (format) type:
                                binary,
                                textRaw (original text),
                                parsedBOW (parsed bag-of-words)
                                parsedText (parsed bag-of-words with positional information)
        :param store_type: a type of the storage engine: mapdb, lucene
                            MapDB is faster than Lucene,  but Lucene uses compression and is,
                            therefore, more memory efficient
        :param max_num_rec: an optional number for the maximum number of records to index.
        :param expected_qty: an expected number of entries in the index, this is just a hint and
                             the default hint value seems to work just fine.
    """
    data_info: InputDataInfo = get_index_query_data_dirs(input_data_dir)
    if not data_info.index_subdirs or data_info.data_file_name is None:
        raise Exception(f'No indexable data is found in {input_data_dir}')
    create_forward_index_explicit_subdirs(input_data_dir=input_data_dir,
                                          subdir_list=data_info.index_subdirs,
                                          output_dir_name=output_dir_name,
                                          index_field_name=index_field_name,
                                          index_type=index_type,
                                          store_type=store_type,
                                          field_type=field_type,
                                          datafile_name=data_info.data_file_name,
                                          max_num_rec=max_num_rec,
                                          expected_qty=expected_qty)


def create_lucene_index_explicit_subdirs(input_data_dir,
                                subdir_list,
                                output_dir_name,
                                index_field_name,
                                exact_match=False,
                                datafile_name=ANSWER_FILE_JSONL_GZ,
                                max_num_rec=MAX_INT32):
    """Call a Java function to create a Lucene index for an explicit list of sub-directories containing data files.

        :param input_data_dir: a root folder containing processed input data.
        :param subdir_list: an array of subdirectories: must not contain commas!
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param exact_match: if true, we create a regular string index, which can be used
                            to retrieve documents using secondary IDs.
        :param datafile_name: the name of the input data file.
        :param max_num_rec: an optional number for the maximum number of records to index.
    """
    assert subdir_list, 'No input directories are given!'
    assert datafile_name is not None, 'Indexable file name should not be None'
    for dn in subdir_list:
        assert not ',' in dn, 'sub-directory names must not have commas!'
    JLuceneIndexer.createLuceneIndex(input_data_dir,
                                     ','.join(subdir_list), datafile_name,
                                     output_dir_name,
                                     index_field_name, exact_match,
                                     max_num_rec)


def create_forward_index_explicit_subdirs(input_data_dir,
                                    subdir_list,
                                    output_dir_name,
                                    index_field_name,
                                    index_type,
                                    field_type,
                                    store_type=STORE_TYPE_MAPDB,
                                    datafile_name=ANSWER_FILE_JSONL_GZ,
                                    max_num_rec=MAX_INT32,
                                    expected_qty=EXPECTED_INDEX_QTY):
    """Call a Java function to create a forward index.

        :param input_data_dir: a root folder containing processed input data.
        :param subdir_list: an array of subdirectories: must not contain commas!
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param index_type: a type of the index: inmem, dataDict, offsetDict
        :param field_type: a field (format) type:
                                binary,
                                textRaw (original text),
                                parsedBOW (parsed bag-of-words)
                                parsedText (parsed bag-of-words with positional information)
        :param store_type: a type of the storage engine: mapdb, lucene
                            MapDB is faster than Lucene,  but Lucene uses compression and is,
                            therefore, more memory efficient.
        :param datafile_name: the name of the input data file.
        :param max_num_rec: an optional number for the maximum number of records to index.
        :param expected_qty: an expected number of entries in the index, this is just a hint and
                             the default hint value seems to work just fine.
    """
    assert subdir_list, 'No input directories are given!'
    assert datafile_name is not None, 'Indexable file name should not be None'
    for dn in subdir_list:
        assert not ',' in dn, 'sub-directory names must not have commas!'

    i_index_type = JForwardIndex.getIndexType(index_type)
    assert i_index_type.toString() != UNKNOWN_TYPE, f'Incorrect index type: {index_type}'

    i_store_type = JForwardIndex.getStoreType(store_type)
    assert i_store_type.toString() != UNKNOWN_TYPE, f'Incorrect store type: {store_type}'

    i_field_type = JForwardIndex.getIndexFieldType(field_type)
    assert i_field_type.toString() != UNKNOWN_TYPE, f'Incorrect field type: {field_type}'

    # A slightly hacky way to clean-up the previous index, which makes certain assumptions
    # about naming of the files. Better of course, delegate this to the Java level eventually
    os.makedirs(output_dir_name, exist_ok=True)
    for fn in os.listdir(output_dir_name):
        if fn == index_field_name or fn.startswith(f'{index_field_name}.'):
            full_path_fn = os.path.join(output_dir_name, fn)
            # Index contents can be stored in sub-directories (Lucene backend does it)
            if os.path.exists(full_path_fn):
                if os.path.isdir(full_path_fn):
                    shutil.rmtree(full_path_fn)
                else:
                    os.unlink(full_path_fn)

    JBuildFwdIndexApp.createForwardIndex(input_data_dir,
                                         ','.join(subdir_list), datafile_name,
                                         output_dir_name,
                                         index_field_name,
                                         i_index_type, i_store_type, i_field_type,
                                         max_num_rec, expected_qty)


