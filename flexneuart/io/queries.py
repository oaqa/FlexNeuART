import json
import os

from flexneuart.io.utils import jsonl_gen
from flexneuart.config import QUESTION_FILE_JSON, DOCID_FIELD
from flexneuart.io import FileWrapper

def is_json_query_file(file_name):
    """Checks if the input is a JSONL query file (using name only)."""
    return os.path.split(file_name)[1] == QUESTION_FILE_JSON


def read_queries(file_name):
    """Read queries from a JSONL file and checks the document ID is set.

    :param file_name: an input file name
    :return: an array where each entry is a parsed query JSON.
    """
    return list(jsonl_gen(file_name))


def read_queries_dict(file_name):
    """Read queries from a JSONL file and checks the document ID is set.

    :param file_name: an input file name
    :return: an dictionary where keys are query IDs and values are parsed query JSONs.
    """
    return {e[DOCID_FIELD] : e for e in jsonl_gen(file_name) }


def write_queries(query_list, file_name):
    """Write queries to a JSONL file.

    :param query_list: an array of parsed JSON query entries
    :param file_name: an output file
    """
    with FileWrapper(file_name, 'w') as f:
        for e in query_list:
            f.write(json.dumps(e))
            f.write('\n')


def write_queries_dict(query_dict : dict, file_name):
    """Write queries stored in the form of a dictionary (query ids as keys).

    :param query_dict: query dictionary: keys are query IDs.
    :param file_name: an output file
    """
    write_queries(list(query_dict.values()), file_name)

