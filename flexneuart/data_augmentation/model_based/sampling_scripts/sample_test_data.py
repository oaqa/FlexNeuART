from collections import Counter

data_path = "/home/ubuntu/efs/capstone/data_aug/data/msmarco/derived_data/{0}"
test_orig = data_path.format("cedr_lucene_index_text_1000_100_0_10_0_s0_bitext/text_raw/test_run.txt")

def read_data_as_dict(data_file):
    orig_dict = {}
    with open(data_file) as f:
        for line in f:
            qid = line[:-1].split()[0]
            if qid not in orig_dict:
                orig_dict[qid] = [line[:-1]]
            else:
                orig_dict[qid].append(line[:-1])
    return orig_dict

orig_dict = read_data_as_dict(test_orig)

counts = []
for k, v in orig_dict.items():
    counts.append(len(v))