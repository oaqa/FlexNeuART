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
import sys
import json

from flexneuart.io import open_with_default_enc


def get_num_datapoint_annotated(json_dataset):
    return sum([len(json_dataset[i]) for i in list(json_dataset.keys())])


def load_json(filename):
    with open_with_default_enc(filename, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    filename = sys.argv[1]
    print(get_num_datapoint_annotated(load_json(filename)))
