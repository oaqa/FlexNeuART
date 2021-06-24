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

"""Misc FlexNeuART utils."""
from jnius import autoclass

JHashMap = autoclass('java.util.HashMap')


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
