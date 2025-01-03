#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", dest="cfg_file", help="config file", default="CornerNetLine", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=50000, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument('--cache_path', dest="cache_path", type=str)
    parser.add_argument('--result_path', dest="result_path", type=str)
    parser.add_argument('--tar_data_path', dest="tar_data_path", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_dir", dest="data_dir", default="c:/work/linedata(1023)", type=str)

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(db, split, testiter, debug=False, suffix=None):
    with torch.no_grad():
        result_dir = system_configs.result_dir
        result_dir = os.path.join(result_dir, str(testiter), split)

        if suffix is not None:
            result_dir = os.path.join(result_dir, suffix)

        make_dirs([result_dir])

        test_iter = system_configs.max_iter if testiter is None else testiter
        print("loading parameters at iteration: {}".format(test_iter))

        print("building neural network...")
        nnet = NetworkFactory(db)
        print("loading parameters...")
        nnet.load_params(test_iter)

        from testfile.test_line_cls_pure_real import testing
        path = 'testfile.test_bar_pure'


        testing = importlib.import_module(path).testing
        nnet.cuda()
        nnet.eval_mode()
        testing(db, nnet, result_dir, debug=debug)

if __name__ == "__main__":

    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = args.cfg_file
    configs["system"]["data_dir"] = args.data_dir
    print(f"cache: ", args.cache_path)
    configs["system"]["cache_dir"] = args.cache_path
    configs["system"]["result_dir"] = args.result_path
    configs["system"]["tar_data_dir"] = args.tar_data_path
    system_configs.update_config(configs["system"])

    # print("system config...")
    # pprint.pprint(system_configs.full)

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    test_split  = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]


    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    testing_db = datasets[dataset](configs["db"], split)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    print("we are here")
    test(testing_db, args.split, args.testiter, args.debug, args.suffix)
