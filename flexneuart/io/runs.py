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
from tqdm import tqdm
from flexneuart.io.utils import FileWrapper
from flexneuart.io import open_with_default_enc
from typing import List, Tuple

FAKE_RUN_ID = "fake_run"


def gen_run_entry_str(query_id, doc_id, rank, score, run_id):
    """A simple function to generate one run entry.

    :param query_id: query id
    :param doc_id:   document id
    :param rank:    entry rank
    :param score:   entry score
    :param run_id:   run id

    """
    return f'{query_id} Q0 {doc_id} {rank} {score} {run_id}'


def read_run_dict(file_name):
    """Read a run file in the form of a dictionary where keys are query IDs.

    :param file_name: run file name
    :return:
    """
    result = {}
    with FileWrapper(file_name) as f:
        for ln, line in enumerate(tqdm(f, desc='loading run (by line)', leave=False)):
            line = line.strip()
            if not line:
                continue
            fld = line.split()
            if len(fld) != 6:
                ln += 1
                raise Exception(
                    f'Invalid line {ln} in run file {file_name} expected 6 white-space separated fields by got: {line}')

            qid, _, docid, rank, score, _ = fld
            result.setdefault(qid, {})[docid] = float(score)

    return result


def get_sorted_scores_from_score_dict(query_run_dict) -> List[Tuple[str, float]]:
    """Take a dictionary of document scores indexed by the document id
    and produce a list of (document id, score tuples) sorted
    in the order of decreasing scores.

    :param   query_run_dict: a single-query run info in the dictionary format.
    :return  a list of pairs (document ID, document socre) sorted by the score in the decreasing order.
    """
    # items() are pairs document ID, document score, but the sorting uses the pair:
    # document score, document ID
    return list(sorted(query_run_dict.items(), key=lambda x: (x[1], x[0]), reverse=True))


def write_run_dict(run_dict, file_name, run_id=FAKE_RUN_ID):
    """Write a dictionary-stored run to a file. The input
       is actually a dictionary of dictinoary. The outer
       dictionary is a set of query-specific results
       indexed by the query id. And the internal dictionary
       is a set of document scores indexed by the document id.
       Before writing data, it is resorted within each query.

    :param run_dict:   a run dictionary
    :param file_name:  an output file name
    :param run_id:     a run ID to use
    """
    with FileWrapper(file_name, 'w') as runfile:
        for qid in run_dict:
            scores = get_sorted_scores_from_score_dict(run_dict[qid])
            for i, (did, score) in enumerate(scores):
                runfile.write(gen_run_entry_str(qid, did, i + 1, score, run_id) + '\n')

