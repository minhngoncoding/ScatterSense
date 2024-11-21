import os
import json
import torch
import pprint
import argparse
import matplotlib
import numpy as np
from test_pipeline import methods

matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBar
from RuleGroup.LineQuiry import GroupQuiry
from RuleGroup.LIneMatch import GroupLine
from RuleGroup.Pie import GroupPie
import math
from PIL import Image, ImageDraw, ImageFont
torch.backends.cudnn.benchmark = False
import requests
import time
import re
from icecream import ic

torch.cuda.empty_cache()
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet Bar")
    parser.add_argument("--cfg_file", dest="cfg_file", help="config file", default="CornerNetPureBar", type=str)
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


def load_net(testiter, cfg_name, data_dir, cache_dir, result_dir, cuda_id=0):
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = result_dir
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet


def Pre_load_nets():
    methods = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", "data/clsdata(1031)", "data/clsdata(1031)/cache",
                                "data/clsdata(1031)/result")

    from testfile.test_line_cls_pure_real import testing
    path = 'testfile.test_%s' % "CornerNetCls"
    testing_cls = importlib.import_module(path).testing
    methods['Cls'] = [db_cls, nnet_cls, testing_cls]
    db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", "data/bardata(1031)", "data/bardata(1031)/cache",
                                "data/bardata(1031)/result")
    path = 'testfile.test_%s' % "CornerNetPureBar"
    testing_bar = importlib.import_module(path).testing
    methods['Bar'] = [db_bar, nnet_bar, testing_bar]

    return methods

def test(image_path, debug=False, suffix=None, min_value=None, max_value=None):
    image_name  = image_path.split("/")[-1]
    image_cls  = Image.open(image_path)
    image = cv2.imread(image_path)

    print("Predicted as BarChart")

    with torch.no_grad():
        results = methods['Cls'][2](image, methods['Cls'][0], methods['Cls'][1], debug=False)
        info = results[0]
        tls = results[1]
        brs = results[2]

        image_painted, cls_info = GroupCls(image_cls, tls, brs)
        # ic(results)
        # ic(info)
        # ic(tls)
        # ic(brs)
        ic(image_painted)
        numpy_image = np.array(image_painted)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)
        # cv2.imshow("Image", numpy_image)
        # cv2.waitKey(0)


        results = methods['Bar'][2](image, methods['Bar'][0], methods['Bar'][1], debug=False)
        tls = results[0]
        brs = results[1]
        ic(cls_info)

        if 5 in cls_info.keys():
            plot_area = cls_info[5][0:4]

        print("Grouping Bar")
        image_painted, bar_data = GroupBar(image_painted, tls, brs, plot_area, min_value=None, max_value=None)

        print("Saving Image")
        image_painted.save(f"test_result/{image_name}")

        return


if __name__ == "__main__":
    methods = Pre_load_nets()
    target_path = "test_img"

    for image_path in os.listdir(target_path):
        if image_path.split(".")[-1] not in ["png", "jpg", "jpeg"]:
            continue
        image_path = os.path.join(target_path, image_path)
        test(image_path)

