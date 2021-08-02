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
import sys
import argparse
import json

# These parameter names must match parameter names in config.sh
EXTR_TYPE_FINAL_PARAM = "extrTypeFinal"
EXPER_SUBDIR_PARAM = "experSubdir"
TEST_ONLY_PARAM = "testOnly"
MODEL_FINAL_PARAM = "modelFinal"

FEAT_EXPER_SUBDIR = "feat_exper"

REL_DESC_PATH_PARAM = 'rel_desc_path'
OUT_DIR_PARAM = 'outdir'


class BaseParser:
    def init_add_args(self):
        pass

    def __init__(self, prog_name):
        self.parser = argparse.ArgumentParser(description=prog_name)
        self.parser.add_argument('--' + OUT_DIR_PARAM, metavar='output directory',
                                 help='output directory',
                                 type=str, required=True)
        self.parser.add_argument('--' + REL_DESC_PATH_PARAM, metavar='relative descriptor path',
                                 help='relative descriptor path',
                                 type=str, required=True)
        self.parser.add_argument('--exper_subdir', metavar='exper. results subdir.',
                                 help='top-level sub-directory to store experimental results',
                                 type=str, default=FEAT_EXPER_SUBDIR)
        self.init_add_args()

    def get_args(self):
        """
        :return: argument objects, to be used
        """
        return self.args

    def parse_args(self):
        """This is deliberately implemented with a delayed optimization,
        so that a user can add new parameter definitions before arguments
        are parsed.
        """
        self.args = self.parser.parse_args()
        print(self.args)


def gen_rerank_descriptors(args, extr_json_gen_func, json_desc_name, json_sub_dir):
    """
    A generic function to write a bunch of experimental descrptors (for the re-ranking only scenario).

    :param args:              arguments previously produce by the class inherited from BaseParser
    :param extr_json_gen_func:   generator of extractor JSON and its file ID.
    :param json_desc_name:      the name of the top-level descriptor file that reference individual extractor JSONs.
    :param json_sub_dir:        a sub-directory to store extractor JSONs.

    """
    desc_data_json = []

    args_var = vars(args)

    out_json_sub_dir = os.path.join(args_var[OUT_DIR_PARAM], json_sub_dir)
    if not os.path.exists(out_json_sub_dir):
        os.makedirs(out_json_sub_dir)

    for file_id, json_desc, test_only, model_final in extr_json_gen_func():
        json_file_name = file_id + '.json'

        desc = {EXPER_SUBDIR_PARAM: os.path.join(args.exper_subdir, json_sub_dir, file_id),
                EXTR_TYPE_FINAL_PARAM: os.path.join(args_var[REL_DESC_PATH_PARAM], json_sub_dir, json_file_name),
                TEST_ONLY_PARAM: int(test_only)}
        if model_final is not None:
            desc[MODEL_FINAL_PARAM] = model_final

        desc_data_json.append(desc)

        with open(os.path.join(out_json_sub_dir, json_file_name), 'w') as of:
            json.dump(json_desc, of, indent=2)

    with open(os.path.join(args.outdir, json_desc_name), 'w') as of:
        json.dump(desc_data_json, of, indent=2)
