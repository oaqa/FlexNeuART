import argparse
import time
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from torch.utils.data import Dataset, DataLoader
import csv
import logging
import datetime
from tqdm import tqdm
import multiprocessing as mp
import random
import os

USE_LOGGING = False

def setup_logging():
    USE_LOGGING = True
    logging.basicConfig(filename='in_pars.log', filemode='a', level=logging.DEBUG)
    logging.info("Start Logging at - {0}".format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

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
                if self.max_examples and counter==self.max_examples:
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


class GPTNeoInitialization:
    def __init__(self):
        pass

    def tokenizer(self):
        return GPT2Tokenizer

    def base_model(self):
        return GPTNeoForCausalLM

class BloomInitialization:
    def __init__(self):
           pass
    
    def tokenizer(self):
        return BloomTokenizerFast
    
    def base_model(self):
        return BloomForCausalLM


class ModelInitialization:

    def __init__(self, model_name:str):
        self.model_name = model_name
        self.model_map = { "EleutherAI/gpt-neo-125M": GPTNeoInitialization,
                           "EleutherAI/gpt-neo-1.3B": GPTNeoInitialization,
                           "EleutherAI/gpt-neo-2.7B": GPTNeoInitialization,
                           "bloom" : BloomInitialization }

    def initialize_model(self):
        initilizer_class = self.model_map[self.model_name]()
        return initilizer_class.tokenizer(), initilizer_class.base_model()

def write_to_output(syn_query_list,syn_query_probs, did_list , aug_query_tsv_op, aug_query_qrels_op, timestamp, counter):
    try:
        with open(aug_query_tsv_op, 'a') as aug_query_tsv_op, open(aug_query_qrels_op, 'a') as aug_query_qrels_op:
            tsv_writer = csv.writer(aug_query_tsv_op, delimiter='\t')
            for query_text, query_prob, doc_id in zip(syn_query_list, syn_query_probs, did_list):
                new_qid = "QP" + str(timestamp) + '_' + str(counter)

                tsv_writer.writerow(["query", new_qid, query_text, query_prob])

                line = new_qid + " 0 " + str(doc_id) + " 1\n"
                aug_query_qrels_op.write(line)

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

    tokenizer_class, model_class = ModelInitialization(args.engine).initialize_model()
    tokenizer = tokenizer_class.from_pretrained(args.engine)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    question_index = []
    vocab_size = len(tokenizer.get_vocab())
    for tid in range(vocab_size):
        token_text = tokenizer.decode([tid])
        if '?' in token_text:
            question_index.append(tid)
    
    model = model_class.from_pretrained(args.engine, return_dict_in_generate=True)
    model = model.to(device)

    inpars_collater = InParsCollater(tokenizer)

    inpars_loader = DataLoader(inpars_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=inpars_collater)

    start_time = time.time()
    for i, batch in enumerate(tqdm(inpars_loader)):
        torch.cuda.empty_cache()
        input_data = batch[1]['input_ids'].to(device=next(model.parameters()).device)
        with torch.no_grad():
            model_out = model.generate(input_data,
                do_sample=True,
                max_new_tokens=args.max_tokens,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.batch_decode(model_out["sequences"])

        try:
            final_queries, final_probs = postprocess_queries(gen_text, batch[2],model_out, question_index)
            query_id_counter = write_to_output(final_queries,
                                final_probs,
                                batch[0], 
                                args.aug_query+split_num,
                                args.aug_query_qrels+split_num,
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
    qrels = ""
    queries = ""

    for i in range(num_splits):
        with open(args.aug_query+str(i)) as q:
            queries += q.read()
        with open(args.aug_query_qrels+str(i)) as q:
            qrels += q.read()
    
    with open(args.aug_query, "a") as op:
        op.write(queries)
    
    with open(args.aug_query_qrels, "a") as op:
        op.write(qrels)

def remove_splits(args, num_splits):

    for i in range(num_splits):
        os.remove(args.aug_query+str(i))
        os.remove(args.aug_query_qrels+str(i))
        

def main(args):

    inpars_dataset = InParsDataset(args.original_doc, args.prompt_template, args.max_examples)

    num_gpus_available = torch.cuda.device_count()

    if args.num_gpu==1 or args.num_gpu==0:
        setup_logging()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        generate_queries(args, inpars_dataset, device, "0")
    
    else:
        gpus_to_use = args.num_gpu
        if num_gpus_available < gpus_to_use:
            print("{0} GPU's not available, running using {1} GPU's".format(gpus_to_use, num_gpus_available))
            gpus_to_use = num_gpus_available
        
        # split data into gpus_to_use parts 
        dataset_split_size = len(inpars_dataset) // gpus_to_use
        splits = [dataset_split_size for _ in range(gpus_to_use)]
        splits[-1] += (len(inpars_dataset)%gpus_to_use)

        data_splits = torch.utils.data.random_split(inpars_dataset, splits)
        generation_args = [(args, data_splits[i], "cuda:{0}".format(i), str(i)) for i in range(gpus_to_use)]

        with mp.Pool(gpus_to_use) as p:
            p.starmap(generate_queries, generation_args)
            p.close()
            p.join()
        
        collate_output_files(args, gpus_to_use)
        remove_splits(args, gpus_to_use)
    
    print("Generation Done")


if __name__ == '__main__':
    # use :
    # python3 generate_queries_gptneo.py --original_doc /home/ubuntu/output.csv --aug_query aug_query.tsv --aug_query_qrels aug_query_qrels.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_doc', type=str, required=True,
                        help='Full Path to TSV file containing document IDs and texts')
    parser.add_argument('--aug_query', type=str, required=True,
                        help='Full Path of TSV file where generated query and their new IDs will be written')
    parser.add_argument('--aug_query_qrels', type=str, required=True,
                        help='Full Path of .txt file to store the query and document ID relation pairs')
    parser.add_argument('--engine', type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument('--prompt_template', nargs="*", default=['prompts/vanilla_prompt.txt'])
    parser.add_argument('--max_examples', type=int, default=10,
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

    args = parser.parse_args()

    main(args)
