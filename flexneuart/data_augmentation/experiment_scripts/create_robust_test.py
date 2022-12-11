import os
import argparse
from flexneuart.data_augmentation.augmentation_module import DataAugmentModule
import shutil

def perform_checks(args):
    if os.path.exists(args.input_dir)==False:
        raise ValueError("Input path does not exist")
    if os.path.exists(args.output_dir):
        raise ValueError("Output path exists.")
    else:
        os.system("mkdir -p {0}".format(args.output_dir))
    
    expected_files = ["data_docs.tsv", "data_query.tsv", "qrels.txt", "test_run.txt", "train_pairs.tsv"]
    files = os.listdir(args.input_dir)

    for f in files:
        if f not in expected_files:
            raise ValueError("File {0} not found in input directory.".format(f))


def augment_queries(dma, args):
    query_path = os.path.join(args.input_dir, "data_query.tsv")
    test_file = os.path.join(args.input_dir, "test_run.txt")
    test_query = []
    with open(test_file) as f:
        for line in f:
            query_id = line.strip().split()[0]
            test_query.append(query_id)
    test_query = set(test_query)

    visit = {}
    with open(query_path) as f:
        for line in f:
            _, query_id, text = line.strip().split("\t")
            if query_id in test_query and query_id not in visit:
                visit[query_id], _ = dma.augment(text, "")
            else:
                visit[query_id] = text.strip()
    
    op_file = os.path.join(args.output_dir, "data_query.tsv")
    with open(op_file, "w") as f:
        for k, v in visit.items():
            f.write('\t'.join(["query", k, v]))
            f.write('\n')


def augment_docs(dma, args):
    doc_path = os.path.join(args.input_dir, "data_docs.tsv")
    test_docs = []
    test_file = os.path.join(args.input_dir, "test_run.txt")
    with open(test_file) as f:
        for line in f:
            doc_id = line.strip().split()[2]
            test_docs.append(doc_id)
    test_docs = set(test_docs)

    visit = {}
    with open(doc_path) as f:
        for line in f:
            _, doc_id, text = line.strip().split("\t")
            if doc_id in test_docs and doc_id not in visit:
                _, visit[doc_id] = dma.augment("", text)
            else:
                visit[doc_id] = text.strip()
    
    op_file = os.path.join(args.output_dir, "data_docs.tsv")
    with open(op_file, "w") as f:
        for k, v in visit.items():
            f.write('\t'.join(["doc", k, v]))
            f.write('\n')


def copy_files(args):
    os.system("cp {0} {1}".format(os.path.join(args.input_dir, "test_run.txt"), 
                                os.path.join(args.output_dir, "test_run.txt")))
    
    os.system("cp {0} {1}".format(os.path.join(args.input_dir, "qrels.txt"), 
                                os.path.join(args.output_dir, "qrels.txt")))
    
    os.system("cp {0} {1}".format(os.path.join(args.input_dir, "train_pairs.tsv"), 
                                os.path.join(args.output_dir, "train_pairs.tsv")))

def main(args):
    dma = DataAugmentModule(args.da_techniques, args.da_conf, args.da_prob)

    if args.query:
        augment_queries(dma, args)
        os.system("cp {0} {1}".format(os.path.join(args.input_dir, "data_docs.tsv"), 
                                os.path.join(args.output_dir, "data_docs.tsv")))
    elif args.docs:
        augment_docs(dma, args)
        os.system("cp {0} {1}".format(os.path.join(args.input_dir, "data_query.tsv"), 
                                os.path.join(args.output_dir, "data_query.tsv")))
    else:
        augment_docs(dma, args)
        augment_queries(dma, args)
    
    copy_files(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help='Path to input directory with FlexNeuART data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory where output files will be written')
    parser.add_argument('--da_conf', type=str, required=True,
                        help='Path to augmentation config path')
    parser.add_argument('--da_prob', type=float, required=True,
                        help='Augmentation probability to be used while perturbing test data')
    parser.add_argument('--da_techniques', nargs='*', metavar='Data Augmentation Methods',
                        help='List of Data augmentation methods')
    
    noise_type_group = parser.add_mutually_exclusive_group(required=True)
    noise_type_group.add_argument('--query', action='store_true', help='Augment Queries')
    noise_type_group.add_argument('--document', action='store_true', help='Augment Documents')
    noise_type_group.add_argument('--both', action='store_true', help='Augment both Queries and Documents')

    args = parser.parse_args()

    perform_checks(args)
    main(args)