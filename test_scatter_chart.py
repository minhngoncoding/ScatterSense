import os
import json
import torch
import pprint
import argparse
import matplotlib
import numpy as np
# from test_pipeline import methods
# from RuleGroup.Scatter import GroupScatter

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
from RuleGroup.LineQuiry import GroupQuiryRaw
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
    parser = argparse.ArgumentParser(description="Test CornerNet Scatter")
    parser.add_argument(
        "--cfg_file",
        dest="cfg_file",
        help="config file",
        default="CornerNetScatter",
        type=str,
    )
    parser.add_argument(
        "--testiter",
        dest="testiter",
        help="test at iteration i",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--split",
        dest="split",
        help="which split to use",
        default="validation",
        type=str,
    )
    parser.add_argument("--cache_path", dest="cache_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--tar_data_path", dest="tar_data_path", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--data_dir", dest="data_dir", default="c:/work/linedata(1023)", type=str
    )

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

    split = {"training": train_split, "validation": val_split, "testing": test_split}[
        "validation"
    ]

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

    from testfile.test_line_cls_pure_real import testing

    db_scatter, nnet_scatter = load_net(
        10000,
        "CornerNetScatter",
        "data/scatterdata",
        "data/scatterdata/cache",
        "data/scatterdata/result",
    )
    path = "testfile.test_%s" % "CornerNetScatter"
    testing_scatter = importlib.import_module(path).testing
    methods["Scatter"] = [db_scatter, nnet_scatter, testing_scatter]

    return methods


def test(image_path, debug=False, suffix=None, min_value=None, max_value=None):
    image_name = image_path.split("/")[-1]
    image_dir = os.path.dirname(image_path)

    image_cls = Image.open(image_path)
    image = cv2.imread(image_path)

    print("Predicted as ScatterChart")

    with torch.no_grad():
        numpy_image = np.array(image)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)
        # cv2.imshow("Image", numpy_image)
        # cv2.waitKey(0)

        results = methods["Scatter"][2](
            image, methods["Scatter"][0], methods["Scatter"][1], debug=False
        )
        keys = results[0]
        hybrids = results[1]

        # print("keys", keys)
        image_painted, quiry, keys, hybrids = GroupQuiryRaw(image, keys, hybrids)

        for key in keys:
            bbox = key["bbox"]
            x, y = int(bbox[0]), int(bbox[1])
            cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        for key in hybrids:
            bbox = key["bbox"]
            x, y = int(bbox[0]), int(bbox[1])
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        if not os.path.exists(f"{image_dir}/results"):
            os.makedirs(f"{image_dir}/results")

        output_path = f"{image_dir}/results/{image_name}"
        cv2.imwrite(output_path, image)
        print(f"Saving Image at {output_path}")
        print("Hybrids", hybrids)
        # image_painted, bar_data = GroupScatter(image_painted, tls, brs, plot_area, min_value=None, max_value=None)
        if not os.path.exists(f"{image_dir}/results"):
            os.makedirs(f"{image_dir}/results")
        #
        print(f"Saving Image at {image_dir}/results")
        # image_painted.save(f"{image_dir}/results/{image_name}")
        # cv2.imshow("Image", image_painted)
        # Save the image
        cv2.imwrite(f"{image_dir}/results/{image_name}", image_painted)
        return


if __name__ == "__main__":
    methods = Pre_load_nets()
    target_path = "data/scatterdata/scatter/images/test"

    for image_path in os.listdir(target_path):
        if image_path.split(".")[-1] not in ["png", "jpg", "jpeg"]:
            continue
        image_path = os.path.join(target_path, image_path)
        test(image_path)
