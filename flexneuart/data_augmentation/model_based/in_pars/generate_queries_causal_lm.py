#!/usr/bin/env python
import argparse
import time
import torch
import json
from logging.handlers import QueueHandler, RotatingFileHandler

from flexneuart.models.train.amp import get_amp_processors
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import Dataset, DataLoader
import csv
import logging
import datetime
from tqdm import tqdm
import multiprocessing as mp
import random
import os
import pandas as pd
import sys

# function to setup logger in every worker
def worker_configurer(queue):
    h = QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)

# Every worker communicates with listenser for logging
def listener_configurer():
    root = logging.getLogger()
    h = RotatingFileHandler('in_pars.log', 'a')
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)

# Instantiate the listener
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

# Dataset class for loading the dataset to generate queries
# Format expected "doc" \t doc_id \t text
class InParsDataset(Dataset):
    def __init__(self, doc_file, prompt_path, max_examples):
        self.doc_file = doc_file
        self.max_examples = max_examples
        self.get_docs()
        self.prompts, self.prompt_names = self.__read_prompts(prompt_path)
        self.num_prompts = len(self.prompts)

    # Getting max examples
    def get_docs(self):
        
        all_docs_input = list()
        with open(self.doc_file,"r") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            counter = 0
            for i, line in enumerate(tsv_file):
                if self.max_examples is not None and counter >= self.max_examples:
                    break
                if line[0]=="query":
                    continue
                counter+=1
                all_docs_input.append((line[1], line[2].strip()))                
        
        self.sents = all_docs_input
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, index):
        prompt_index = random.randint(0, self.num_prompts-1)
        return self.sents[index][0], self.prompts[prompt_index].format(document_text=self.sents[index][1]), self.sents[index][1], self.prompt_names[prompt_index]

    def __read_prompts(self, prompt_paths):
        prompts = []
        prompt_names = []
        for pp in prompt_paths:
            prompt_file = open(pp)
            prompts.append(prompt_file.read())
            prompt_file.close()
            prompt_names.append(pp)

        return prompts, prompt_names

# implements the collate function to create the input batch
class InParsCollater(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_texts = [i[1] for i in batch]
        batch_ids = [i[0] for i in batch]
        batch_lengths = [len(i) for i in batch_texts]
        orig_data = [i[2] for i in batch]
        batch_data = self.tokenizer.batch_encode_plus(batch_texts, return_tensors='pt', padding=True)
        prompt_name = [i[3] for i in batch]

        return batch_ids, batch_data, batch_lengths, orig_data, prompt_name

# dump the generated queries to file
# output format - query_id \t query \t query_score
def write_to_output(add_query_pref,
                    syn_query_list, 
                    syn_query_probs, did_list, 
                    output,
                    timestamp, counter, doc_data, prompt_names):
    try:
        with open(output, 'a') as outf:
            
            iter_data = zip(syn_query_list, syn_query_probs, did_list, doc_data, prompt_names)
            for query_text, query_prob, doc_id, doc_text, prompt_name in iter_data:
                qid = f'QP{add_query_pref}_{timestamp}_{doc_id}_{counter}'
                query_text = query_text.replace('\n', '').strip()
                json_dict = {'doc_id':  doc_id,
                            'query_id': qid,
                            'doc_text': doc_text,
                            'question': query_text,
                            'log_probs':  [query_prob],
                            'prompt_name': prompt_name}
                outf.write(json.dumps(json_dict, ensure_ascii=False) + '\n')
                counter += 1
        return counter
    except IOError as e:
        print(e)

# utility function that gets the probaility of generated tokens
# from the logit matrix
def gather_2d_on_last_dim(tensor, index, shape):
    
    flattened_tensor = tensor.view(-1, tensor.shape[-1])
    flattened_index = index.view(-1)
    flattened_gathered_tensor = flattened_tensor[
        torch.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.view(shape)

# parse generated logits and text to get scores
def postprocess_queries(generated_texts, prompt_lengths, model_outputs, qindex):

    # extract the first question that appears after the orignial prompt
    questions = [text[ text.find('Example 1:') + prompt_length:] for text, prompt_length in zip(generated_texts,prompt_lengths)]
    q_inds = [q.find("?") for q in questions]
    final_queries = [q[:q_ind+1] for q,q_ind in zip(questions,q_inds)]

    # as mutiple tokens can have '?'
    # find all such tokens
    valid_query_indices = []
    for i, q in enumerate(final_queries):
        if q!="":
            valid_query_indices.append(i) 

    #probs
    probs = torch.stack(model_outputs.scores, dim=1).log_softmax(-1) # batchsize*tokensize*vocabsize

    # model_output["sequences"].shape = [batch_size x output_size]
    # probs.shape = [batch_size x max_new_tokens x vocab_size]
    length_input = model_outputs["sequences"].shape[1] - probs.shape[1]
    output_ids = model_outputs["sequences"][valid_query_indices,length_input:]

    # get the scores of the tokens that were observed in the output
    probs = gather_2d_on_last_dim(probs[valid_query_indices,:,:], output_ids, output_ids.shape)

    # remove all the extra tokens that appear after the '?'
    clip_extra_output_ids = []
    for t in output_ids:
      for qid in qindex:
        try:
          clip_id = (t == qid).nonzero(as_tuple=True)[0][0]
          clip_extra_output_ids.append(clip_id)
          break
        except:
          continue

    clip_extra_output_ids = torch.tensor(clip_extra_output_ids).to(device=probs.device)
    
    # mask out the extra tokens
    index_matrix = torch.arange(probs.shape[1]).expand(len(clip_extra_output_ids), probs.shape[1])
    index_matrix = index_matrix.to(device=probs.device)
    masks = index_matrix < clip_extra_output_ids.unsqueeze(1)
    
    probs = probs * masks

    probs = torch.sum(probs, axis=1)/clip_extra_output_ids

    # initializing -1e9 will ensure very small probs for empty sentences
    final_probs = [-1e9 for _ in range(len(final_queries))]

    for idx,prob in zip(valid_query_indices,probs):
        final_probs[idx] = prob.cpu().item()

    return final_queries, final_probs

# the main function that would generate the queries using the specified model
def generate_queries(args, inpars_dataset, device, split_num, queue, configurer):

    configurer(queue)
    logger = logging.getLogger()

    query_id_timestamp = time.time()
    query_id_counter = 0

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.engine)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token

        question_index = []
        vocab_size = len(tokenizer.get_vocab())
        for tid in range(vocab_size):
            token_text = tokenizer.decode([tid])
            if '?' in token_text:
                question_index.append(tid)

        # It can be much faster to load a model once rather do it again in every process.
        # Unfortunately, this often results in 'too many open files' error :-(
        model = AutoModelForCausalLM.from_pretrained(args.engine, return_dict_in_generate=True)
        if args.amp and device != 'cpu':
            model = model.half()
        model = model.to(device)

        auto_cast_class, _ = get_amp_processors(args.amp)

        inpars_collater = InParsCollater(tokenizer)

        inpars_loader = DataLoader(inpars_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=inpars_collater)

        start_time = time.time()
        loader = inpars_loader
        # Let us TQDM only in a single process
        if split_num == "0":
            loader = tqdm(loader)
        for i, batch in enumerate(loader):
            # torch.cuda.empty_cache()
            input_data = batch[1]['input_ids'].to(device=next(model.parameters()).device)
            with torch.no_grad():
                with auto_cast_class():
                    model_out = model.generate(input_data,
                        do_sample=True,
                        max_new_tokens=args.max_tokens,
                        output_scores=True,
                        pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.batch_decode(model_out["sequences"])

            try:
                final_queries, final_probs = postprocess_queries(gen_text, batch[2], model_out, question_index)
                query_id_counter = write_to_output(args.add_query_pref,
                                    final_queries,
                                    final_probs,
                                    batch[0], 
                                    args.output+split_num,
                                    query_id_timestamp,
                                    query_id_counter,
                                    batch[3], batch[4])
            except Exception as e:
                curr_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                logger.log(logging.DEBUG, "{0} {1}".format(curr_time, str(e)))
                logger.log(logging.DEBUG, "Batch {1} Doc ID's {2}".format(curr_time, i, str(batch[2])))
                continue

        print("Total Time = {0}".format(time.time()-start_time))
        print('Done!')
    except Exception as e:
        logger.log(logging.ERROR, str(e))

# for collating jsonl files simply concat them
def collate_jsonl(args, num_splits):
    queries = ""
    for i in range(num_splits):
        try:
            with open(args.output+str(i)) as q:
                queries += q.read()
        except:
            print("Output corresponding to split number - {0} not found".format(i))
            print("This is probably because a process failed. Check in_pars.log for more details.")
            print("Skipping this split number")
            continue

    with open(args.output, "a") as op:
        op.write(queries)


# remove the segmented files
def remove_splits(args, num_splits):
    for i in range(num_splits):
        if os.path.exists(args.output+str(i)):
            os.remove(args.output+str(i))
        

def main(args):
    queue = mp.Manager().Queue(-1)
    listener = mp.Process(target=listener_process,
                        args=(queue, listener_configurer))
    listener.start()

    inpars_dataset = InParsDataset(args.original_doc, args.prompt_template, args.max_examples)

    num_gpus_available = torch.cuda.device_count()

    worker_configurer(queue)
    logger = logging.getLogger()
    curr_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logger.log(logging.INFO, "Generation Starting at {0}".format(curr_time))

    gpus_to_use = args.num_gpu
    if num_gpus_available < gpus_to_use:
        print("{0} GPU's not available, running using {1} GPU's".format(gpus_to_use, num_gpus_available))
        gpus_to_use = num_gpus_available

    if args.num_gpu==1 or args.num_gpu==0:
        # setup_logging()
        device = args.device if torch.cuda.is_available() else "cpu"
        generation_args = [(args, inpars_dataset, device, "0", queue, worker_configurer)]
    
    else:
        # split data into gpus_to_use parts 
        dataset_split_size = len(inpars_dataset) // gpus_to_use
        splits = [dataset_split_size for _ in range(gpus_to_use)]
        splits[-1] += (len(inpars_dataset)%gpus_to_use)
        
        data_splits = torch.utils.data.random_split(inpars_dataset, splits)
        generation_args = [(args, data_splits[i], "cuda:{0}".format(i), str(i), queue, worker_configurer) 
                                for i in range(gpus_to_use)]

    # Launch Tasks with Porcess Pool
    with mp.Pool(gpus_to_use) as p:
        p.starmap_async(generate_queries, generation_args)
        p.close()
        p.join()

    queue.put_nowait(None)
    listener.join()

    try:
        collate_jsonl(args, gpus_to_use)
        remove_splits(args, gpus_to_use)
    except Exception as e:
        print(e)
        print("Generation Failed. Check in_pars.log for more details")
        sys.exit(1)
    
    print("Generation Done")


if __name__ == '__main__':
    #
    # use :
    #
    # python3 generate_queries_causal_lm.py \
    #             --original_doc /home/ubuntu/output.csv \
    #             --prompt_template prompts/vanilla_prompt.txt
    #             --output output.jsonl \
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_doc', type=str, required=True,
                        help='Full Path to TSV file containing document IDs and texts')
    parser.add_argument('--add_query_pref', type=str, default='', 
                        help='Additional query prefix')
    parser.add_argument('--output', type=str, required=True, 
                        help='An output file (in JSONL format)')
    parser.add_argument('--engine', type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument('--prompt_template', nargs="*", required=True)
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of documents to read from the collection.')
    parser.add_argument('--max_tokens', type=int, default=32, help='Max tokens to be generated.')
    parser.add_argument('--batch_size', type=int, default=12, help="dataloader batch size")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature. Zero means greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_gpu', type=int, default=1,
                        help="Number of GPU's to run inference on. Dataset will be divided")
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use when running the model. cuda:0, cuda:1 and so on')
    parser.add_argument('--amp', action='store_true', help="Use automatic mixed-precision")

    args = parser.parse_args()

    if os.path.exists(args.output):
        print('Output file', args.output, 'already exists!')
        sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    torch.multiprocessing.set_start_method('spawn')

    main(args)
