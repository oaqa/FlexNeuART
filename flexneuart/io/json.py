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
import json

def read_json(file_name):
    """Read and parse JSON file

    :param file_name: JSON file name
    """
    with open(file_name) as f:
        data = f.read()

    return json.loads(data)


def save_json(file_name, data, indent=4):
    """Save JSON data

    :param file_name:   output file name
    :param data:        JSON data
    :param indent:      JSON indentation
    """
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=indent)

