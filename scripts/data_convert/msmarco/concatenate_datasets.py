import os
import re
import json
import argparse


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
    return list(map(lambda x: x.lower(), re.sub('[^a-z_a-Z0-9 ]', ' ', query_text).split()))


if __name__ == "__main__":
    args = get_args()
    print(args)

    with open(args.output, "a") as output_file:
        for datafile in os.listdir(args.input_datapath):
            print('Processing:', datafile)
            if datafile.endswith(".json"):
                filename = os.path.join(args.input_datapath, datafile)
                with open(filename, "r") as f:
                    dataset = json.load(f)
                output = get_converted_data(dataset)
                for datapoint in output:
                    json.dump(datapoint, output_file)
                    output_file.write('\n')
