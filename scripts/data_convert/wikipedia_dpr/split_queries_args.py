import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split queries and corresponding QREL files.')

    parser.add_argument('--src_dir',
                        metavar='input data directory',
                        help='input data directory',
                        type=str, required=True)
    parser.add_argument('--dst_dir',
                        metavar='output data directory',
                        help='output data directory',
                        type=str, required=True)
    parser.add_argument('--seed',
                        metavar='random seed',
                        help='random seed',
                        type=int, default=0)
    parser.add_argument('--partitions_names',
                        metavar='names of partitions to split at',
                        help='names of partitions to split at separated by comma',
                        default="bitext,train_fusion,dev",
                        type=str)
    parser.add_argument('--partitions_sizes',
                        metavar='sizes of partitions to split at',
                        help="sizes (in queries) of partitions to split at separated by comma (one of the values can be -1, "
                             "in that case all left queries go to that partition)",
                        default="-1,10000,10000",
                        type=str)

    return Arguments(parser.parse_args())

class Arguments:
    def __init__(self, raw_args):
        self.raw_args = raw_args

    @property
    def src_dir(self):
        return self.raw_args.src_dir

    @property
    def dst_dir(self):
        return self.raw_args.dst_dir
    
    @property
    def seed(self):
        return self.raw_args.seed
     
    @property    
    def partitions_names(self):
        return self.raw_args.partitions_names.split(',')
    
    def partitions_sizes(self, queries_count):
        raw_values = list(map(int, self.raw_args.partitions_sizes.split(',')))
        nondefined_count = 0
        defined_sum = 0
        for value in raw_values:
            if value != -1:
                assert 0 < value < queries_count
                defined_sum += value
            else:
                nondefined_count += 1

        if nondefined_count == 0 and defined_sum == queries_count:
            return raw_values
        elif nondefined_count == 1 and defined_sum < queries_count:
            raw_values[raw_values.index(-1)] = queries_count - defined_sum
            return raw_values
        else:
            raise ValueError("invalid --partitions_sizes argument")
