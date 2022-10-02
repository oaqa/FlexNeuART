import os
import argparse
from generate_files import generate_files

def main():
    args = parse_arguments()
    arg_check(args)
    # positive query generation arguments

    # negative document generation arguments

    # generate files
    generate_files(args)

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    # FILES
    # input files
    parser.add_argument('--original_doc', type=str, required=True,
                        help='Path to .tsv file containing document IDs and texts')
    parser.add_argument('--original_query', type=str, required=True,
                        help='Path to .tsv file containing query IDs and texts')
    parser.add_argument('--original_qrels', type=str, required=True,
                        help='Path to .txt file containing relations between query and document')

    # output directory
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Path to the output data directory')


    # temp files
    parser.add_argument('--aug_query', type=str, default='temp/aug_query.tsv',
                        help='Path to the .tsv file containing augmented queries')
    parser.add_argument('--aug_query_qrels', type=str, default='temp/aug_query_qrels.txt',
                        help='Path to .txt file containing relations between augmented queries and documents')
    parser.add_argument('--neg_doc', type=str, default='temp/neg_doc.tsv',
                        help='Path to the .tsv file containing negative documents')
    parser.add_argument('--neg_doc_qrels', type=str, default='temp/neg_doc_qrels.tsv',
                        help='Path to .txt file containing relations between augmented queries and negative documents')
    parser.add_argument('--overwrite', action='store_true')

    # positive query generation arguments
    parser.add_argument('--engine', type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument('--prompt_template', type=str, default='prompts/vanilla_prompt.txt')
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

    # negative document generation arguments
    parser.add_argument('--index', type=str, default='lucene_index',
                        help='Type of the index (eg. lucene_index)')
    parser.add_argument('--max_docs_retrieved', type=int, default=100,
                        help='The maximum number of documents to be retrieved')

    # generate data files
    parser.add_argument('--original_weight', type=float, default=1.0)
    parser.add_argument('--aug_query_weight', type=float, default=1.0)
    parser.add_argument('--neg_doc_weight', type=float, default=1.0)

    args = parser.parse_args()
    
    return args


def arg_check(args):
    for file_path in [args.original_doc, args.original_query, args.original_qrels]:
        if not os.path.exists(file_path):
            raise Exception("Input file not found")

    if os.path.exists(args.output_directory):
        if not args.overwrite:
            raise Exception("Output directory exists and can be overwritten. If this is expected, please pass --overwrite")
    else:
        os.makedirs(args.output_directory)

    for temp_file_path in [args.aug_query, args.aug_query_qrels, args.neg_doc, args.neg_doc_qrels]:
        if not args.overwrite and os.path.exists(temp_file_path):
            raise Exception("Temporary file exists and can be overwritten. If this is expected, please pass --overwrite")

    return

if __name__ == '__main__':
    main()
    
