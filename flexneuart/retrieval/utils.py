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
from jnius import autoclass

from flexneuart.config import DOCID_FIELD

JHashMap = autoclass('java.util.HashMap')
DataEntryFields = autoclass('edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields')

def dict_to_hash_map(dict_obj):
    """Convert a Python dictionary to a Java HashMap object. Caution:
       values in the dictionary need to be either simple types, or
       proper Java object references created through jnius autoclass.

    :param dict_obj:   a Python dictionary whose values and keys are either simple types
                       or Java objects creates via jnius autoclass
    :return: a Java HashMap
    """

    res = JHashMap()
    for k, v in dict_obj.items():
        res.put(k, v)

    return res


def query_dict_to_dataentry_fields(query_dict, default_query_id=None):
    """Convert a query dictionary object to an internal frame work object of the type
       DataEntryFields.

       LIMITATION: Currently, no binary values or string arrays are supported.

    :param query_dict:          query key-value dictionary that may or may not have the query/doc ID
    :param default_query_id:    a default query ID to use if query_dict has none.

    :return: an instance of the type DataEntryFields.
    """
    if DOCID_FIELD in query_dict:
        qid = query_dict[DOCID_FIELD]
    else:
        qid = default_query_id
        if qid is None:
            raise Exception('Specify either a default query ID or provide it as a value of {DOCID_FIELD} field')

    res = DataEntryFields(qid)

    for k, v in query_dict.items():
        if k != DOCID_FIELD:
            if type(v) == str:
                res.setString(str(k), v)
            elif type(v) == int:
                res.setInteger(str(k), v)
            elif type(v) == float:
                res.setFloat(str(k), v)
            else:
                raise Exception('Unsupported type %s, supported types are str, int, float' % str(type(v)))


    return res
