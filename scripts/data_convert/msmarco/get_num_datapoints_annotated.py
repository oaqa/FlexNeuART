import sys
import json


def get_num_datapoint_annotated(json_dataset):
    return sum([len(json_dataset[i]) for i in list(json_dataset.keys())])


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    filename = sys.argv[1]
    print(get_num_datapoint_annotated(load_json(filename)))
