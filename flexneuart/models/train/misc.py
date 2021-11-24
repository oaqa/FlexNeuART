import torch
import gc

from tqdm import tqdm

from flexneuart.models.train.batching import BatchingValidationGroupByQuery, BatchObject

from flexneuart.models.train.amp import *
from flexneuart.config import DEVICE_CPU, TQDM_FILE


def clean_memory(device_name):
    gc.collect()
    if device_name != DEVICE_CPU:
        with torch.cuda.device(device_name):
            torch.cuda.empty_cache()


def run_model(model,
              device_name, batch_size, amp,
              max_query_len, max_doc_len,
              dataset, orig_run,
              cand_score_weight=0,
              desc='valid',
              use_progress_bar=True
              ):
    """Execute model on a given query set: produce document scores.

    :param model:           a model object
    :param device_name:     a device name
    :param batch_size:      validation batch size
    :param amp:             true to enable AMP
    :param max_query_len:   maximum query length
    :param max_doc_len:     maximum document lengths
    :param dataset:         a tuple: query dictionary, document dictionary
    :param orig_run:        a run to re-run
    :param cand_score_weight: a weight of the candidate provider score
    :param use_progress_bar: True to enable he progress bar
    :param desc:            an optional descriptor

    :return: a re-ranked run, where each query document pair is scored using the model (optionally
             fusing the scores with the candidate provider scores)
    """

    auto_cast_class, _ = get_amp_processors(amp)
    rerank_run = {}

    clean_memory(device_name)

    cand_score_weight = torch.FloatTensor([cand_score_weight]).to(device_name)
    if use_progress_bar:
        pbar = tqdm(total=sum(len(r) for r in orig_run.values()), ncols=80, desc=desc, leave=False,  file=TQDM_FILE)
    else:
        pbar = None
    with torch.no_grad():

        model.eval()

        iter_val = BatchingValidationGroupByQuery(batch_size=batch_size,
                                                  dataset=dataset, model=model,
                                                  max_query_len=max_query_len,
                                                  max_doc_len=max_doc_len,
                                                  run=orig_run)
        for batch in iter_val():
            with auto_cast_class():
                batch: BatchObject = batch
                batch.to(device_name)
                model_scores = model(*batch.features)
                assert len(model_scores) == len(batch)
                scores = model_scores + batch.cand_scores * cand_score_weight
                # tolist() works much faster compared to extracting scores one by one using .item()
                scores = scores.tolist()

            for qid, did, score in zip(batch.query_ids, batch.doc_ids, scores):
                rerank_run.setdefault(qid, {})[did] = score
            if pbar is not None:
                pbar.update(len(batch))
                pbar.refresh()

  
    if pbar is not None:
        pbar.close()

    return rerank_run
