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

from flexneuart.config import ANSWER_FILE_JSONL_GZ, MAX_INT32

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

def create_lucene_index_wrapper(input_data_dir,
                                subdir_list,
                                output_dir_name,
                                index_field_name,
                                exact_match=False,
                                datafile_name=ANSWER_FILE_JSONL_GZ,
                                max_num_rec=MAX_INT32):
    """Call a Java function to create a Lucene index.

        :param input_data_dir: a root folder containing processed input data.
        :param subdir_list: an array of subdirectories: must not contain commas!
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param exact_match: if true, we create a regular string index, which can be used
                            to retrieve documents using secondary IDs.
        :param datafile_name: the name of the input data file.
        :param max_num_rec: an optional number for the maximum number of records to index.
    """
    for dn in subdir_list:
        assert not ',' in dn, 'sub-directory names must not have commas!'
    JLuceneIndexer.createLuceneIndex(input_data_dir,
                                     ','.join(subdir_list), datafile_name,
                                     output_dir_name,
                                     index_field_name, exact_match,
                                     max_num_rec)


def create_forward_index_wrapper(input_data_dir,
                                subdir_list,
                                output_dir_name,
                                index_field_name,
                                index_type,
                                store_type,
                                field_type,
                                datafile_name=ANSWER_FILE_JSONL_GZ,
                                max_num_rec=MAX_INT32,
                                expected_qty=EXPECTED_INDEX_QTY):
    """Call a Java function to create a forward index.

        :param input_data_dir: a root folder containing processed input data.
        :param subdir_list: an array of subdirectories: must not contain commas!
        :param output_dir_name: the output directory to store the index
        :param index_field_name: the name of text field to be used for indexing.
        :param exact_match: if true, we create a regular string index, which can be used
                            to retrieve documents using secondary IDs.
        :param index_type: a type of the index: inmem, dataDict, offsetDict
        :param store_type: a type of the storage engine: mapdb, lucene
        :param field_type: an field (format) type:
                                binary,
                                textRaw (original text),
                                parsedBOW (parsed bag-of-words)
                                parsedText (parsed bag-of-words with positional information)
        :param datafile_name: the name of the input data file.
        :param max_num_rec: an optional number for the maximum number of records to index.
        :param expected_qty: an expeced number of entries in the index
    """
    for dn in subdir_list:
        assert not ',' in dn, 'sub-directory names must not have commas!'

    iIndexType = JForwardIndex.getIndexType(index_type)
    assert iIndexType.toString() != UNKNOWN_TYPE, f'Incorrect index type: {index_type}'

    iStoreType = JForwardIndex.getStoreType(store_type)
    assert iStoreType.toString() != UNKNOWN_TYPE, f'Incorrect store type: {store_type}'

    iFieldType = JForwardIndex.getIndexFieldType(field_type)
    assert iFieldType.toString() != UNKNOWN_TYPE, f'Incorrect field type: {field_type}'

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
									     iIndexType, iStoreType, iFieldType,
									     max_num_rec, expected_qty)

