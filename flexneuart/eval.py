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

import numpy as np
import subprocess
import math

from typing import Union, Dict

from flexneuart.io import create_temp_file
from flexneuart.io.runs import get_sorted_scores_from_score_dict, write_run_dict, read_run_dict
from flexneuart.io.qrels import read_qrels_dict, write_qrels_dict

from flexneuart import Registry
eval_registry = Registry()
register = eval_registry.register

DEFAULT_TREC_EVAL_PATH='trec_eval/trec_eval'

FAKE_DOC_ID = "THIS_IS_A_VERY_LONG_FAKE_DOCUMENT_ID_THAT_SHOULD_NOT_MATCH_ANY_REAL_ONES"

METRIC_RECALL_PREF = 'recall'
METRIC_MAP_PREF = 'map'
METRIC_MRR_PREF = 'mrr'
METRIC_MRR_PREF_ADD = 'recip_rank'
METRIC_NDCG_PREF = 'ndcg'

# This mimics trec_eval
CUTOFF_ARR = [5, 10, 15, 20, 30, 100, 200, 500, 1000]

METRIC_NDCG_LIST = [f'{METRIC_NDCG_PREF}@{k}' for k in CUTOFF_ARR]
METRIC_RECALL_LIST = [f'{METRIC_RECALL_PREF}@{k}' for k in CUTOFF_ARR]
METRIC_MRR_LIST = [METRIC_MRR_PREF, METRIC_MRR_PREF_ADD] + [f'{METRIC_MRR_PREF}@{k}' for k in CUTOFF_ARR]
METRIC_MAP_LIST = [METRIC_MAP_PREF] + [f'{METRIC_MAP_PREF}@{k}' for k in CUTOFF_ARR]

LOG2_MULT = math.log(2)

METRIC_LIST = METRIC_MRR_LIST + METRIC_NDCG_LIST + METRIC_MAP_LIST +  METRIC_RECALL_LIST

RELEVANCE_THRESHOLD = 1e-5

qrel_cache = {}

class MetricBase:
    def __init__(self, cut_off):
        self.cut_off = cut_off

    @staticmethod
    def get_total_rel(qrel_dict):
        """Compute a total number of relevant items."""
        return sum([int(rel > RELEVANCE_THRESHOLD) for did, rel in qrel_dict.items()])

    def get_cut_rels(self, rels_arr):
        """
            Conveninence wrapper to cut a list of relevance score with respect
            to pre-defined cutoff value. When there is not cutoff (infinity)
            then the original value is returned
        """
        if type(self.cut_off) == int:
            return rels_arr[0 : self.cut_off]
        else:
            return rels_arr


@register(METRIC_NDCG_PREF)
class NormalizedDiscountedCumulativeGain(MetricBase):
    def __init__(self, cut_off=float('inf')):
        super().__init__(cut_off)

    def _dcg(self, rels_sorted_by_scores):
        res = 0
        for i, rel in enumerate(rels_sorted_by_scores):
            if rel > RELEVANCE_THRESHOLD:
                res += rel / math.log(2 + i)

        return res * LOG2_MULT

    def __call__(self, rels_sorted_by_scores, qrel_dict):
        """
            Calculate NDCG. The function assumes,
            we already sorted everything in the order of decreasing scores.

            :param rels_sorted_by_scores: true relevance judgements sorted by scores.
            :param qrel_dict: true relevance scores indexed by document ids
            :return: NDCG.
        """
        # Note that the ideal ranking is also cut off using the same threshold
        idcg = self._dcg(self.get_cut_rels(sorted(qrel_dict.values(), reverse=True)))
        return self._dcg(self.get_cut_rels(rels_sorted_by_scores)) / idcg if idcg > 0 else 0


@register(METRIC_MAP_PREF)
class MeanAveragePrecision(MetricBase):
    def __init__(self, cut_off=float('inf')):
        super().__init__(cut_off)

    def __call__(self, rels_sorted_by_scores, qrel_dict):
        """
            Calculate mean average precision. The function assumes,
            we already sorted everything in the order of decreasing scores.

            :param rels_sorted_by_scores: true relevance judgements sorted by scores.
            :param qrel_dict: true relevance scores indexed by document ids
            :return: Mean average precision.
        """
        result = 0.
        tot_rel_qty = self.get_total_rel(qrel_dict)
        if tot_rel_qty == 0:
            return 0

        pos = 0
        for i, rel in enumerate(self.get_cut_rels(rels_sorted_by_scores)):
            if rel > RELEVANCE_THRESHOLD:
                pos += 1.
                result += pos / (i + 1.)

        return result / tot_rel_qty


@register([METRIC_MRR_PREF, METRIC_MRR_PREF_ADD])
class MeanReciprocalRank(MetricBase):
    def __init__(self, cut_off=float('inf')):
        super().__init__(cut_off)
        
    def __call__(self, rels_sorted_by_scores, qrel_dict):
        for i, rel in enumerate(self.get_cut_rels(rels_sorted_by_scores)):
            if rel > RELEVANCE_THRESHOLD:
                return 1 / (i + 1.)
        return 0


@register(METRIC_RECALL_PREF)
class RecallAtK(MetricBase):
    def __init__(self, cut_off=float('inf')):
        super().__init__(cut_off)

    def __call__(self, rels_sorted_by_scores, qrel_dict):
        pos = 0

        tot_rel_qty = self.get_total_rel(qrel_dict)
        if tot_rel_qty == 0:
            return 0

        for i, rel in enumerate(self.get_cut_rels(rels_sorted_by_scores)):
            if rel > RELEVANCE_THRESHOLD:
                pos += 1

        return pos / tot_rel_qty


def get_cutoff(prefix: str, desc_with_cutoff) -> Union[int, float]:
    """A function to extract the cutoff value from metric names such as :

            ndcg@20, mrr@10, mrr, ndcg, etc ...

       When @K is missing, the cutoff value is infinite.

    :param prefix:   metric name prefix (but without @)
    :param desc_with_cutoff: a metric name description, e.g., mrr@10, ndcg@20, or mrr

    :return: an integer cutoff value or float('inf')
    """
    assert desc_with_cutoff.startswith(prefix), \
        f'Description {desc_with_cutoff} should have prefix: {prefix} or {prefix}@'
    if prefix == desc_with_cutoff:
        return float('inf')
    cutoff_str = desc_with_cutoff[len(prefix) + 1:]
    try:
        res = int(cutoff_str)

        assert res > 0, f'Cutoff value {cutoff_str} is not a positive integer!'

        return res
    except ValueError as e:
        raise Exception(f'Cutoff value {cutoff_str} is not a positive integer!')


def create_metric_obj(desc_with_cutoff : str):
    """A function that creates a metric evaluation object from metric description.

    :param desc_with_cutoff: a metric name description, e.g., mrr@10, ndcg@20, or mrr
    :return: a tuple: a rank cutoff vlue, an object that can compute evaluation metric
    """
    prefix_list = []
    for prefix, func in eval_registry.registered.items():
        prefix_list.append(prefix)
        if desc_with_cutoff.startswith(prefix):
            return func(get_cutoff(prefix, desc_with_cutoff))

    raise Exception(f'Unsupported metric {desc_with_cutoff}, supported metrics (can optionally add @K): '
                    + ','.join(prefix_list))


def get_eval_results(use_external_eval : bool,
                     eval_metric : str,
                     run : Union[str, Dict[str, Dict[str, int]]],
                     qrels : Union[str, Dict[str, Dict[str, int]]],
                     ret_query_vals=False,
                     trec_eval_path : str = DEFAULT_TREC_EVAL_PATH):
    """
        Carry out internal or external evaluation.

        :param use_external_eval:  True to use external evaluation tools.
        :param eval_metric:        Evaluation metric description.
        :param run:                a run file name or a run file dictionary
        :param qrels:              a QREL file name or a QREL dictionary
        :param ret_query_vals:     return query-specific values (not just averages)
        :param trec_eval_path:     A path to a trec_eval binary
        :param run_file:           A run file to store results (or None).

        :return:  if :ret_query_vals is False return average metric value
                  if :ret_query_vals is True return a  tuple (the average metric value, a dictionary of query-specific values)
    """
    if use_external_eval:
        if eval_metric == METRIC_MAP_PREF:
            m = 'map'
        elif eval_metric.startswith(METRIC_MAP_PREF) or \
             eval_metric.startswith(METRIC_NDCG_PREF):
            m = eval_metric.replace('@', '_cut_')
        elif eval_metric.startswith(METRIC_RECALL_PREF):
            m = eval_metric.replace('@', '_')
        elif eval_metric in [METRIC_MRR_PREF, METRIC_MRR_PREF_ADD]:
            m = 'recip_rank'
        else:
            raise Exception('Unsupported trec_eval metric: ' + eval_metric)

        full_res = trec_eval(run, qrels, m, trec_eval_path=trec_eval_path)
    else:
        f = create_metric_obj(eval_metric)
        full_res = internal_eval(run=run, qrels=qrels, metric_func=f)

    if ret_query_vals:
        return full_res
    else:
        return full_res[0]


def trec_eval(run : Union[str, Dict[str, Dict[str, int]]],
              qrels : Union[str, Dict[str, Dict[str, int]]],
              metric : str,
              trec_eval_path : str = DEFAULT_TREC_EVAL_PATH
              ):
    """
        Run an external tool: trec_eval and retrieve results.

        :param run:    a run file name or a run file dictionary
        :param qrels:   a QREL file name or a QREL dictionary
        :param metric:  a metric code (should match what trec_eval prints)
        :param trec_eval_path: a path to a trec_eval binary

        :return: a tuple (the average metric value, np array of query-specific values)
    """
    unlink_files = []
    if type(run) != str:
        run_f = create_temp_file()
        unlink_files.append(run_f)
        write_run_dict(run, run_f)
    else:
        run_f = run

    if type(qrels) != str:
        qrel_f = create_temp_file()
        unlink_files.append(qrel_f)
        write_qrels_dict(qrels, qrel_f)
    else:
        qrel_f = qrels

    try:
        trec_eval_params = [trec_eval_path,
                            '-q',  # all query results
                            '-m', 'all_trec',
                            qrel_f, run_f]
        res_dict = {}
        avg_res = None
        seen_metric = False
        metric_set = set()
        for line in subprocess.check_output(trec_eval_params).decode().split('\n'):
            if not line:
                continue
            fields = line.rstrip().split()
            metric_set.add(fields[0])
            if len(fields) >= 3 and fields[0] == metric:
                seen_metric = True
                qid = str(fields[1])
                val = float(fields[2])
                if fields[1] != 'all':
                    res_dict[qid] = val
                else:
                    avg_res = val
    except Exception as e:
        for fn in unlink_files:
            os.unlink(fn)
        raise e

    for fn in unlink_files:
        os.unlink(fn)

    assert seen_metric, f'Wrong metric: {metric}, supported metrics: ' + ', '.join(list(metric_set))
    assert avg_res is not None, 'Some inconsistency detected: no aggregate/average value is produced by trec_eval!'

    return avg_res, res_dict


def internal_eval(run : Union[str, Dict[str, Dict[str, int]]],
             qrels : Union[str, Dict[str, Dict[str, int]]],
             metric_func,
             debug=False):
    """
        Evaluate run stored in a file using QRELs stored in a file.

        :param run:    a run file name or a run file dictionary
        :param qrels:   a QREL file name or a QREL dictionary
        :param metric_func:    a metric function or class instance with overloaded __call__

        :return: a tuple (the average metric value, a dictionary of query-specific values)
    """
    if type(run) == str:
        run_dict = read_run_dict(run)
    else:
        run_dict = run

    if type(qrels) == str:
        qrel_dict = read_qrels_dict(qrels)
    else:
        qrel_dict = qrels

    res_dict = {}

    for qid, score_dict in run_dict.items():
        # trec_eval ignores queries without QRELs
        if qid not in qrel_dict:
            continue
        rels_sorted_by_scores = []

        val = 0

        if qid in qrel_dict:
            query_qrel_dict = qrel_dict[qid]

            for did, score in get_sorted_scores_from_score_dict(score_dict):
                rel_score = 0
                if did in query_qrel_dict:
                    rel_score = query_qrel_dict[did]

                rels_sorted_by_scores.append(rel_score)

            val = metric_func(rels_sorted_by_scores, query_qrel_dict) if query_qrel_dict else 0

        if debug:
            print('%s %g' % (qid, val))

        res_dict[qid] = val

    avg = np.mean(list(res_dict.values()))
    if debug:
        print('mean %g' % avg)

    return avg, res_dict

