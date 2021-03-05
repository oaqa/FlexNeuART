#!/usr/bin/python3

import sys
import re
import json

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

    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    with open(input_filepath, "r") as f:
        dataset = json.load(f)
    
    datapoints = get_converted_data(dataset)

    with open(output_filepath, "a") as f:
        for datapoint in datapoints:
            json.dump(datapoint, f)
            f.write('\n')


    