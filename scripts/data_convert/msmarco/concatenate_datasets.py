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
import os
import re
import json
import argparse

from flexneuart.io import open_with_default_enc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_datapath")
    parser.add_argument("--output")
    return parser.parse_args()

def get_converted_data(dataset):
    datapoints = []
    for key in dataset["query_id"].keys():
        datapoints.append({"query_id": dataset["query_id"][key],
                           "query": dataset["query"][key],
                           "query_type": dataset["query_type"][key],
                           "query_tokens": get_tokens(dataset["query"][key])})

    return datapoints


def get_tokens(query_text):
    return list(map(lambda x: x.lower(), re.sub('[^a-zA-Z0-9 ]', ' ', query_text).split()))


if __name__ == "__main__":
    args = get_args()
    print(args)

    with open_with_default_enc(args.output, "a") as output_file:
        for datafile in os.listdir(args.input_datapath):
            print('Processing:', datafile)
            if datafile.endswith(".json"):
                filename = os.path.join(args.input_datapath, datafile)
                with open_with_default_enc(filename, "r") as f:
                    dataset = json.load(f)
                output = get_converted_data(dataset)
                for datapoint in output:
                    json.dump(datapoint, output_file)
                    output_file.write('\n')
