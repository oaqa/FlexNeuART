import json
import argparse
import csv
import os
import sys

def extract_docs(input_path, output_path):

    file = open(input_path)
    output_file = open(output_path, "w")
    tsv_writer = csv.writer(output_file, delimiter='\t')

    for json_line in file:
        line = json.loads(json_line)
        tsv_writer.writerow(['doc', line['doc_id'], line['doc_text']])
    
    file.close()
    output_file.close()

    print("Data Extacted !!!!")
    
def perform_checks(input_path, output_path):
    output_dir, output_file = os.path.split(output_path)

    if os.path.isfile(input_path)==False:
        print("Input is not a file or Path does not exist")
        sys.exit(1)
    
    if output_file=='':
        print("Output file name not specified !!")
        sys.exit(1)
    
    if os.path.exists(output_dir)==False and len(output_dir)>0:
        os.makedirs(output_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                    help="Path to the input jsonl file")
    parser.add_argument('--output_path', required=True,
                    help="Path to the output file")

    args = parser.parse_args()

    perform_checks(args.input_path, args.output_path)

    extract_docs(args.input_path, args.output_path)


    

