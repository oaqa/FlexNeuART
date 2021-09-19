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
import numpy as np
import subprocess
import math

from flexneuart.io.runs import get_sorted_scores_from_score_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict


FAKE_DOC_ID = "THIS_IS_A_VERY_LONG_FAKE_DOCUMENT_ID_THAT_SHOULD_NOT_MATCH_ANY_REAL_ONES"

METRIC_MAP = 'map'
# We hardcode 20, b/c it's hardcoded in gdeval.pl
NDCG_TOP_K = 20
METRIC_NDCG20 = 'ndcg@20'
METRIC_MRR = "recip_rank"

METRIC_LIST = [METRIC_MAP, METRIC_NDCG20, METRIC_MRR]

RELEVANCE_THRESHOLD = 1e-5

qrel_cache = {}

class NormalizedDiscountedCumulativeGain:
    def __init__(self, k):
        self._k = k

    def _dcg(self, rels_sorted_by_scores):

        res = 0
        for i, rel in enumerate(rels_sorted_by_scores):
            if i >= self._k:
                break
            if rel > RELEVANCE_THRESHOLD:
                res += (math.pow(2., rel) - 1.) / math.log(2. + i)

        return res

    def __call__(self, rels_sorted_by_scores, qrel_dict):
        """
            Calculate NDCG. The function assumes,
            we already sorted everything in the order of decreasing scores.

            :param rels_sorted_by_scores: true relevance judgements sorted by scores.
            :param qrel_dict: true relevance scores indexed by document ids
            :return: NDCG.
        """
        idcg = self._dcg(sorted(qrel_dict.values(), reverse=True))
        return self._dcg(rels_sorted_by_scores) / idcg if idcg > 0 else 0


class MeanAveragePrecision:
    def __call__(self, rels_sorted_by_scores, qrel_dict):
        """
            Calculate mean average precision. The function assumes,
            we already sorted everything in the order of decreasing scores.

            :param rels_sorted_by_scores: true relevance judgements sorted by scores.
            :param qrel_dict: true relevance scores indexed by document ids
            :return: Mean average precision.
        """
        result = 0.
        post_qty = sum([int(rel > RELEVANCE_THRESHOLD) for did, rel in qrel_dict.items()])

        pos = 0
        for i, rel in enumerate(rels_sorted_by_scores):
            if rel > RELEVANCE_THRESHOLD:
                pos += 1.
                result += pos / (i + 1.)

        return result / post_qty


class MeanReciprocalRank:
    def __call__(self, rels_sorted_by_scores, qrel_dict):
        for i, rel in enumerate(rels_sorted_by_scores):
            if rel > RELEVANCE_THRESHOLD:
                return 1 / (i + 1.)
        return 0


def eval_run(rerank_run, qrels_dict, metric_func, debug=False):
    """
        Evaluate run stored in a file using QRELs stored in a file.

        :param rerank_run:     a run dictionary (of dictionaries)
        :param qrels_dict:     a QRELs dictionary read by the function read_qrels_dict
        :param metric_func:    a metric function or class instance with overloaded __call__

        :return:  the average metric value
    """
    res_arr = []

    assert type(qrels_dict) == dict, \
        "Relevance info object must be a dictionary, make sure you used read_qrels_dict and not read_qrels!"

    for qid, score_dict in rerank_run.items():
        rels_sorted_by_scores = []

        val = 0

        if qid in qrels_dict:
            query_qrel_dict = qrels_dict[qid]

            for did, score in get_sorted_scores_from_score_dict(score_dict):
                rel_score = 0
                if did in query_qrel_dict:
                    rel_score = query_qrel_dict[did]

                rels_sorted_by_scores.append(rel_score)

            val = metric_func(rels_sorted_by_scores, query_qrel_dict) if query_qrel_dict else 0

        if debug:
            print('%s %g' % (qid, val))

        res_arr.append(val)

    res = np.mean(res_arr)
    if debug:
        print('mean %g' % res)

    return res


def get_eval_results(use_external_eval,
                   eval_metric,
                   rerank_run,
                   qrel_file,
                   run_file=None,
                   use_qrel_cache=False):
    """
        Carry out internal or external evaluation.

        :param use_external_eval:   True to use external evaluation tools.
        :param eval_metric:        Evaluation metric (from the METRIC_LIST above)
        :param run_file:           A run file to store results (or None).
        :param qrel_file:          A QREL file.
        :param use_qrel_cache:  use global QREL file cache (dangerous option: there should
                              be no file-name collisions to for this)

        :return:  average metric value.
    """

    if use_external_eval:
        m = None
        if eval_metric == METRIC_MAP:
            m = 'map'
        elif eval_metric == METRIC_NDCG20:
            m = 'ndcg_cut_20'
        elif eval_metric == METRIC_MRR:
            m = 'recip_rank'
        else:
            raise Exception('Unsupported metric: ' + eval_metric)

        assert run_file is not None, "Run file name should not be None"
        write_run_dict(rerank_run, run_file)

        return trec_eval(run_file, qrel_file, m)
    else:
        f = None
        if eval_metric == METRIC_MAP:
            f = MeanAveragePrecision()
        elif eval_metric == METRIC_NDCG20:
            f = NormalizedDiscountedCumulativeGain(NDCG_TOP_K)
        elif eval_metric == METRIC_MRR:
            f = MeanReciprocalRank()
        else:
            raise Exception('Unsupported metric: ' + eval_metric)

        if run_file is not None:
            write_run_dict(rerank_run, run_file)

        global qrel_cache

        if use_qrel_cache and qrel_file in qrel_cache:
            qrels = qrel_cache[qrel_file]
        else:
            qrels = qrel_cache[qrel_file] = read_qrels_dict(qrel_file)

        return eval_run(rerank_run=rerank_run,
                       qrels_dict=qrels,
                       metric_func=f)


def trec_eval(runf, qrelf, metric):
    """
        Run an external tool: trec_eval and retrieve results.

        :param runf:    a run file name
        :param qrelf:   a QREL file name
        :param metric:  a metric code (should match what trec_eval prints)
        :return:
    """
    trec_eval_f = 'trec_eval/trec_eval'
    trec_eval_params = [trec_eval_f,
                        '-m', 'official',
                        '-m', 'ndcg_cut',
                        qrelf, runf]
    # print(' '.join(trec_eval_params))
    for line in subprocess.check_output(trec_eval_params).decode().split('\n'):
        fields = line.rstrip().split()
        if len(fields) == 3 and fields[0] == metric:
            return float(fields[2])

    raise Exception(
        f'Cannot get the value of the metric {metric} by evaluating file {runf} with qrels {qrelf}')

