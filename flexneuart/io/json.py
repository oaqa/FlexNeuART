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

Some functions related to reading/writing & interpreting JSON files.

"""

import json
from flexneuart.io import open_with_default_enc


def read_json(file_name):
    """Read and parse JSON file

    :param file_name: JSON file name
    """
    with open_with_default_enc(file_name) as f:
        data = f.read()

    return json.loads(data)


def save_json(file_name, data, indent=4):
    """Save JSON data

    :param file_name:   output file name
    :param data:        JSON data
    :param indent:      JSON indentation
    """
    with open_with_default_enc(file_name, 'w') as f:
        json.dump(data, f, indent=indent)


def get_val_err_msg_miss(dict_obj, attr_name, allowed_types, attr_default=None):
    """Get a value from the dictionary, produce a (more) readable
       error message when the value is missing or the type is wrong.

    :param dict_obj:       a dictionary of values
    :param attr_name:      key (attribute name)
    :param allowed_types:  allowed types
    :param attr_default:   default value

    :return: value or default (if specified)
    """
    if attr_name in dict_obj:
        val = dict_obj[attr_name]
        for t in allowed_types:
            if type(val) == t:
                return val

        raise Exception(f'Incorrect value type for the field: {attr_name}, expected types: ' +
                        ', '.join([str(t) for t in allowed_types]))
    else:
        if attr_default is not None:
            for t in allowed_types:
                if type(attr_default) == t:
                    return attr_default

                raise Exception(f'Incorrect type of the default value for the field: {attr_name}, expected types: ' +
                                ', '.join([str(t) for t in allowed_types]))

        raise Exception(f'Missing value for field: {attr_name}')


