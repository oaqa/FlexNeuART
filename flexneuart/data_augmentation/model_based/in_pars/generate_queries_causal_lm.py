import argparse
from asyncio.log import logger
from cgitb import handler
import time
import torch

from flexneuart.models.train.amp import get_amp_processors
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import Dataset, DataLoader
import csv
import logging
from logging.handlers import QueueHandler, QueueListener
import datetime
from tqdm import tqdm
import multiprocessing as mp
import random
import os
import pandas as pd

USE_LOGGING = False

def setup_logging():
    USE_LOGGING = True
    logging.basicConfig(filename='in_pars.log', filemode='a', level=logging.DEBUG)
    logging.info("Start Logging at - {0}".format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

def init_worker(q):
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)

def init_logger():
    q = mp.Queue()
    handler = logging.StreamHandler()
    handler.setFormatter("%(levelname)s: %(asctime)s - %(porcess)s - %(message)s")

    ql = QueueListener(q, handler)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return ql, q

class InParsDataset(Dataset):
    def __init__(self, doc_file, prompt_path, max_examples):
        self.doc_file = doc_file
        self.max_examples = max_examples
        self.get_docs()
        self.prompts = self.__read_prompts(prompt_path)
        self.num_prompts = len(self.prompts)

    # Getting max examples
    def get_docs(self):
        
        all_docs_input = list()
        with open(self.doc_file,"r") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            counter = 0
            for i,line in enumerate(tsv_file):
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
        return self.sents[index][0], self.prompts[prompt_index].format(document_text=self.sents[index][1])

    def __read_prompts(self, prompt_paths):
        prompts = []
        for pp in prompt_paths:
            prompt_file = open(pp)
            prompts.append(prompt_file.read())
            prompt_file.close()

        return prompts

class InParsCollater(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_texts = [i[1] for i in batch]
        batch_ids = [i[0] for i in batch]
        batch_lengths = [len(i) for i in batch_texts]
        batch_data = self.tokenizer.batch_encode_plus(batch_texts, return_tensors='pt',padding=True)

        return batch_ids, batch_data, batch_lengths


def write_to_output(syn_query_list, syn_query_probs, did_list, aug_query_tsv_op, timestamp, counter):
    try:
        with open(aug_query_tsv_op, 'a') as aug_query_tsv_op:
            tsv_writer = csv.writer(aug_query_tsv_op, delimiter='\t')
            for query_text, query_prob, doc_id in zip(syn_query_list, syn_query_probs, did_list):
                new_qid = 'QP' + str(timestamp) + f'_{doc_id}_{counter}'
                query_text = query_text.replace('\n', '')
                tsv_writer.writerow(["query", new_qid, query_text, query_prob])

                counter += 1
        return counter
    except IOError as e:
        print(e)

def gather_2d_on_last_dim(tensor, index, shape):
    
    flattened_tensor = tensor.view(-1, tensor.shape[-1])
    flattened_index = index.view(-1)
    flattened_gathered_tensor = flattened_tensor[
        torch.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.view(shape)

def postprocess_queries(generated_texts, prompt_lengths, model_outputs, qindex):

    #suppose we have the model_output
    
    questions = [text[ text.find('Example 1:') + prompt_length:] for text, prompt_length in zip(generated_texts,prompt_lengths)]
    q_inds = [q.find("?") for q in questions]
    final_queries = [q[:q_ind+1] for q,q_ind in zip(questions,q_inds)]

    # valid_query_indices = [i if q!="" for i,q in enumerate(final_queries)]
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

    probs = gather_2d_on_last_dim(probs[valid_query_indices,:,:], output_ids, output_ids.shape)

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

def generate_queries(args, inpars_dataset, device, split_num):
    query_id_timestamp = time.time()
    query_id_counter = 0

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
            query_id_counter = write_to_output(final_queries,
                                final_probs,
                                batch[0], 
                                args.aug_query+split_num,
                                query_id_timestamp,
                                query_id_counter)
        except Exception as e:
            curr_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if USE_LOGGING==True:
                logging.debug("{0} {1}".format(curr_time, str(e)))
                logging.debug("Batch {1} Doc ID's {2}".format(curr_time, i, str(batch[2])))
            else:
                print("{0} {1}".format(curr_time, str(e)))
                print("Batch {1} Doc ID's {2}".format(curr_time, i, str(batch[2])))
            continue
    print("Total Time = {0}".format(time.time()-start_time))

    print('Done!')

def collate_output_files(args, num_splits):    
    df_list = []
    col_names = ['name', 'query_id', 'query_text', 'score']
    for i in range(num_splits):
        df_list.append(pd.read_csv(args.aug_query+str(i), delimiter='\t',
                                    names=col_names, header=None))
    
    df = pd.concat(df_list)

    if args.topk:
        num_to_select = args.topk
    else:
        num_to_select = int(args.topp * len(df))
    
    df.sort_values(by=['score'], ascending=False, inplace=True)

    df.to_csv(args.aug_query, header=False, index=False, sep='\t')

    top_df = df.head(num_to_select).copy()
    top_df['doc_id'] = top_df.apply(lambda row: row['query_id'].split('_')[-2], axis=1)
    top_df['temp_col'] = 0
    top_df['value'] = 1
    top_df = top_df[['query_id', 'temp_col', 'doc_id', 'value']]

    top_df.to_csv(args.aug_query_qrels, header=None, index=None, sep=' ')

def remove_splits(args, num_splits):

    for i in range(num_splits):
        os.remove(args.aug_query+str(i))
        

def main(args):

    inpars_dataset = InParsDataset(args.original_doc, args.prompt_template, args.max_examples)

    num_gpus_available = torch.cuda.device_count()

    gpus_to_use = args.num_gpu
    if num_gpus_available < gpus_to_use:
        print("{0} GPU's not available, running using {1} GPU's".format(gpus_to_use, num_gpus_available))
        gpus_to_use = num_gpus_available

    

    if args.num_gpu==1 or args.num_gpu==0:
        # setup_logging()
        device = args.device if torch.cuda.is_available() else "cpu"
        generate_queries(args, inpars_dataset, device, "0")
    
    else:
        torch.multiprocessing.set_start_method('spawn')

        # split data into gpus_to_use parts 
        dataset_split_size = len(inpars_dataset) // gpus_to_use
        splits = [dataset_split_size for _ in range(gpus_to_use)]
        splits[-1] += (len(inpars_dataset)%gpus_to_use)
        

        data_splits = torch.utils.data.random_split(inpars_dataset, splits)
        generation_args = [(args, data_splits[i], "cuda:{0}".format(i), str(i)) for i in range(gpus_to_use)]

        q_listener, q = init_logger()

        with mp.Pool(gpus_to_use, init_worker, [q]) as p:
            p.starmap_async(generate_queries, generation_args)
            p.close()
            p.join()
            q_listener.stop()
        
    collate_output_files(args, gpus_to_use)
    remove_splits(args, gpus_to_use)
    
    print("Generation Done")


if __name__ == '__main__':
    #
    # use :
    #
    # python3 generate_queries_causal_lm.py \
    #             --original_doc /home/ubuntu/output.csv \
    #             --aug_query aug_query.tsv \
    #             --aug_query_qrels aug_query_qrels.txt
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_doc', type=str, required=True,
                        help='Full Path to TSV file containing document IDs and texts')
    parser.add_argument('--aug_query', type=str, required=True,
                        help='Full Path of TSV file where generated query and their new IDs will be written')
    parser.add_argument('--aug_query_qrels', type=str, required=True,
                        help='Full Path of .txt file to store the query and document ID relation pairs')
    parser.add_argument('--engine', type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument('--prompt_template', nargs="*", default=['prompts/vanilla_prompt.txt'])
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of documents to read from the collection.')
    parser.add_argument('--max_tokens', type=int, default=32, help='Max tokens to be generated.')
    parser.add_argument('--batch_size', type=int, default=12, help="dataloader batch size")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature. Zero means greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--sleep_time', type=float, default=1.5, 
                        help='Time to wait between API calls, in seconds.')        
    parser.add_argument('--include_doc_probs', action='store_true',
                        help='Wheter or not to save the tokens probabilities produeced by the model.')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help="Number of GPU's to run inference on. Dataset will be divided")
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use when running the model. cuda:0, cuda:1 and so on')
    parser.add_argument('--amp', action='store_true', help="Use automatic mixed-precision")

    top_selection_group = parser.add_mutually_exclusive_group(required=True)
    top_selection_group.add_argument('--topk', type=int,
                                    help="selects the K queries with the highest log prob. score")
    top_selection_group.add_argument('--topp', type=float,
                                    help="selects the top p% of queries with highest log prob score")

    args = parser.parse_args()

    main(args)
