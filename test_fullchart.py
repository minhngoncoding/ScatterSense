#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import matplotlib

matplotlib.use("Agg")
import cv2
import importlib
import math
import requests
import time
import re

from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBar
from RuleGroup.LineQuiry import GroupQuiry
from RuleGroup.LIneMatch import GroupLine
from RuleGroup.Box import GroupBox
from icecream import ic
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageDraw, ImageFont

torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument(
        "--cfg_file",
        dest="cfg_file",
        help="config file",
        default="CornerNetLine",
        type=str,
    )
    parser.add_argument(
        "--testiter",
        dest="testiter",
        help="test at iteration i",
        default=50000,
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
        "--data_dir", dest="data_dir", default="data/linedata(1028)", type=str
    )
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        default="C:/work/linedata(1028)/line/images/test2019/f4b5dac780890c2ca9f43c3fe4cc991a_d3d3LmVwc2lsb24uaW5zZWUuZnIJMTk1LjEwMS4yNTEuMTM2.xls-3-0.png",
        type=str,
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

    # Classification Model
    db_cls, nnet_cls = load_net(
        50000,
        "CornerNetCls",
        "data/clsdata(1031)",
        "data/clsdata(1031)/cache",
        "data/clsdata(1031)/result",
    )

    from testfile.test_line_cls_pure_real import testing

    path = "testfile.test_%s" % "CornerNetCls"
    testing_cls = importlib.import_module(path).testing
    methods["Cls"] = [db_cls, nnet_cls, testing_cls]

    # Bar-Line Model
    db_bar, nnet_bar = load_net(
        50000,
        "CornerNetPureBar",
        "data/bardata(1031)",
        "data/bardata(1031)/cache",
        "data/bardata(1031)/result",
    )
    path = "testfile.test_%s" % "CornerNetPureBar"
    testing_bar = importlib.import_module(path).testing
    methods["Bar"] = [db_bar, nnet_bar, testing_bar]

    # Scatter Model
    db_scatter, nnet_scatter = load_net(
        50000,
        "CornerNetScatter",
        "data/scatterdata",
        "data/scatterdata/cache",
        "data/scatterdata/result",
    )
    path = "testfile.test_%s" % "CornerNetScatter"
    testing_scatter = importlib.import_module(path).testing
    methods["Scatter"] = [db_scatter, nnet_scatter, testing_scatter]

    # Box Model
    db_box, nnet_box = load_net(
        50000,
        "CornerNetVerticalBox",
        "data/boxdata",
        "data/boxdata/cache",
        "data/boxdata/result",
    )
    path = "testfile.test_%s" % "CornerNetVerticalBox"
    testing_box = importlib.import_module(path).testing
    methods["Box"] = [db_box, nnet_box, testing_box]
    return methods


methods = Pre_load_nets()


def ocr_result(image_path):
    from dotenv import load_dotenv
    load_dotenv()

    endpoint = os.environ["AZURE_DOC_INTEL_ENDPOINT"]
    key = os.environ["AZURE_DOC_INTEL_KEY"]

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=f,
            content_type="application/octet-stream"
        )
    result = poller.result()

    print("🧾 OCR Result:")
    word_infos = []
    for block in result.pages[0].lines:
        word_infos.append({
            "text": block.content,
            "boundingBox": block.polygon  # already 8 floats: [x0, y0, ..., x3, y3]
    })

    return word_infos

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + (box2[2] - box2[0]) > max(box2[2], box1[2]) - min(
        box2[0], box1[0]
    ) and (box1[3] - box1[1]) + (box2[3] - box2[1]) > max(box2[3], box1[3]) - min(
        box2[1], box1[1]
    ):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2 - Xc1) * (Yc2 - Yc1)
        return intersection_area / ((box2[3] - box2[1]) * (box2[2] - box2[0]))
    else:
        return 0


def get_min_max_value(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    # word_infos = ocr_result(image_path)

    word_infos = [{'boundingBox': [49.0, 9.0, 277.0, 9.0, 277.0, 21.0, 49.0, 20.0],
                  'text': 'NN 50N Nnnnnnnnn Nnnnnn nn Nnnnnnnn'},
                 {'boundingBox': [37.0, 31.0, 46.0, 31.0, 46.0, 38.0, 37.0, 38.0],
                  'text': '64'},
                 {'boundingBox': [37.0, 44.0, 46.0, 44.0, 46.0, 53.0, 36.0, 53.0],
                  'text': '62'},
                 {'boundingBox': [37.0, 70.0, 46.0, 70.0, 46.0, 78.0, 36.0, 79.0],
                  'text': '58'},
                 {'boundingBox': [9.0, 117.0, 10.0, 60.0, 18.0, 60.0, 17.0, 117.0],
                  'text': 'Nnnnnnnnnn'},
                 {'boundingBox': [36.0, 83.0, 50.0, 84.0, 50.0, 92.0, 36.0, 92.0],
                  'text': '55'},
                 {'boundingBox': [37.0, 96.0, 46.0, 96.0, 46.0, 105.0, 37.0, 105.0],
                  'text': '54'},
                 {'boundingBox': [37.0, 109.0, 46.0, 110.0, 47.0, 118.0, 37.0, 118.0],
                  'text': '52'},
                 {'boundingBox': [37.0, 123.0, 46.0, 124.0, 46.0, 131.0, 37.0, 131.0],
                  'text': '50'},
                 {'boundingBox': [88.0, 144.0, 111.0, 145.0, 111.0, 154.0, 87.0, 153.0],
                  'text': 'Ealing'},
                 {'boundingBox': [184.0, 144.0, 210.0, 145.0, 210.0, 153.0, 184.0, 152.0],
                  'text': 'London'},
                 {'boundingBox': [279.0, 144.0, 311.0, 144.0, 311.0, 154.0, 279.0, 154.0],
                  'text': 'England'},
                 {'boundingBox': [25.0, 169.0, 59.0, 169.0, 59.0, 180.0, 25.0, 180.0],
                  'text': '02008'},
                 {'boundingBox': [84.0, 170.0, 117.0, 169.0, 117.0, 179.0, 84.0, 180.0],
                  'text': '2009'},
                 {'boundingBox': [182.0, 166.0, 204.0, 167.0, 204.0, 175.0, 182.0, 174.0],
                  'text': 'Nnnn'}]
    
    for id in title_list:
        if id in cls_info.keys():
            """
            0: Legend
            1: ValueAxisTitle
            2: ChartTitle
            3: CategoryAxisTitle
            4: Plot Area
            5: Inner Plot Area
            """
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [
                    word_info["boundingBox"][0],
                    word_info["boundingBox"][1],
                    word_info["boundingBox"][4],
                    word_info["boundingBox"][5],
                ]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    ic(word_info["text"])                  
                    words.append([word_info["text"], word_bbox[0], word_bbox[1]])
                    word_infos.remove(word_info)
            words.sort(key=lambda x: x[1] + 10 * x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + " "
            title2string[id] = word_string
    
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [
                word_info["boundingBox"][0],
                word_info["boundingBox"][1],
                word_info["boundingBox"][4],
                word_info["boundingBox"][5],
            ]
            word_text = word_info["text"]
            word_text = re.sub("[^-+0123456789.]", "", word_text)
            word_text_num = re.sub("[^0123456789]", "", word_text)
            word_text_pure = re.sub("[^0123456789.]", "", word_text)

            ic(word_text_num, word_bbox[2] <= x_board)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board + 10:
                dis2max = math.sqrt(
                    math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2)
                    + math.pow((word_bbox[1] + word_bbox[3]) / 2 - y_max, 2)
                )
                dis2min = math.sqrt(
                    math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2)
                    + math.pow((word_bbox[1] + word_bbox[3]) / 2 - y_min, 2)
                )
                y_mid = (word_bbox[1] + word_bbox[3]) / 2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == "-":
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == "-":
                        min_value = -min_value
        # print(min_value)
        # print(max_value)
        delta_min_max = max_value - min_value
        delta_mark = min_y - max_y + 0.00000000000000000000000000001
        delta_plot_y = y_min - y_max
        delta = delta_min_max / delta_mark
        if abs(min_y - y_min) / delta_plot_y > 0.1:
            print(abs(min_y - y_min) / delta_plot_y)
            print("Predict the lower bar")
            min_value = int(min_value + (min_y - y_min) * delta)

    return title2string, round(min_value, 2), round(max_value, 2)


def test(
    image_path,
    debug=False,
    suffix=None,
    min_value_official=None,
    max_value_official=None,
    chart_type=None
):
    image_cls = Image.open(image_path)
    image = cv2.imread(image_path)
    with torch.no_grad():
        results = methods["Cls"][2](
            image, methods["Cls"][0], methods["Cls"][1], debug=False
        )
        info = results[0]
        tls = results[1]
        brs = results[2]
        plot_area = []
        image_painted, cls_info = GroupCls(image_cls, tls, brs)

        title2string, min_value, max_value = get_min_max_value(image_path, cls_info)
        
        if min_value_official is not None:
            min_value = min_value_official
            max_value = max_value_official
        chartinfo = [info["data_type"], cls_info, title2string, min_value, max_value]

        if chart_type == "Bar":
            print("Predicted as BarChart")
            results = methods["Bar"][2](
                image, methods["Bar"][0], methods["Bar"][1], debug=False
            )
            tls = results[0]
            brs = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            image_painted, bar_data = GroupBar(
                image_painted, tls, brs, plot_area, min_value, max_value
            )

            return plot_area, image_painted, bar_data, chartinfo

        if chart_type == "VerticalBox":
            print("Predicted as Vertical Box Chart")
            results = methods["Box"][2](
                image, methods["Box"][0], methods["Box"][1], debug=False
            )
            keys = results[0]
            hybrids = results[1]
            image_painted, box_data = GroupBox(image_painted, cens, keys)
            return plot_area, image_painted, box_data, chartinfo
        
        if chart_type == "Scatter":
            print("Predicted as Scatter Chart")
            results = methods["Scatter"][2](
                image, methods["Scatter"][0], methods["Scatter"][1], debug=False, cuda_id=1
            )
            keys = results[0]
            hybrids = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            return plot_area, image_painted, line_data, chartinfo


if __name__ == "__main__":
    tar_path = "test_images"
    images = os.listdir(tar_path)
    from random import shuffle
    shuffle(images)
    for image in tqdm(images):
        if image == "results":
            continue
        path = os.path.join(tar_path, image)
        plot_area, image_painted, line_data, chartinfo = test(path)
        if not os.path.exists(f"{tar_path}/results"):
            os.makedirs(f"{tar_path}/results")
        image_painted.save(f"{tar_path}/results/{image}")
        ic(line_data)
