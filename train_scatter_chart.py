import os

from PIL.ImageOps import expand

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback
from socket import error as SocketError
import errno
import re

from icecream import ic
from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from azureml.core.run import Run
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
from torchviz import make_dot

import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = " expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument(
        "--cfg_file",
        dest="cfg_file",
        help="config file",
        default="CornerNetScatter",
        type=str,
    )
    parser.add_argument(
        "--iter", dest="start_iter", help="train at iteration i", default=0, type=int
    )
    parser.add_argument("--threads", dest="threads", default=1, type=int)
    parser.add_argument(
        "--data_dir", dest="data_dir", default="./data/scatterdata", type=str
    )
    parser.add_argument("--cache_path", dest="cache_path", default="", type=str)
    args = parser.parse_args()
    return args


def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            print("We met some errors!")
            traceback.print_exc()
            continue


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        try:
            data = data_queue.get()
            data["xs"] = [x.pin_memory() for x in data["xs"]]
            data["ys"] = [y.pin_memory() for y in data["ys"]]

            pinned_data_queue.put(data)

            if sema.acquire(blocking=False):
                return
        except SocketError as e:
            if e.errno != errno.ECONNRESET:
                raise
            pass


def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [
        Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs
    ]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks


def export_model_to_onnx(model, sample_data):
    # sample, ind = sample_data(training_dbs[0], 0, data_aug=True)
    images, key_tags, key_tags_grouped, tag_group_lens = sample_data["xs"]

    onnx_input = (images, key_tags, key_tags_grouped, tag_group_lens)

    torch.onnx.export(
        model,
        onnx_input,
        "model.onnx",
        output_names=["output"],
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )


def train(training_dbs, validation_db, start_iter):
    learning_rate = system_configs.learning_rate
    max_iteration = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot = system_configs.snapshot
    val_iter = system_configs.val_iter
    display = system_configs.display
    decay_rate = system_configs.decay_rate
    stepsize = system_configs.stepsize
    val_ind = 0

    print("Building model...")

    # Load Network model from "nnet/py_factory.py" -> "models/CornerNetScatter.py"
    nnet = NetworkFactory(training_dbs[0])

    # Export the model to ONNX
    training_size = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue = Queue(32)
    # queues storing pinned data for training
    pinned_training_queue = queue.Queue(32)

    # load data sampling function from sample/scatter.py (augmented data)
    data_file = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data
    print(f"data sampling {sample_data}")

    print("=" * 50)
    print("start parallel reading...")
    print("=" * 50)

    training_tasks = init_parallel_jobs(
        training_dbs, training_queue, sample_data, data_aug=True
    )
    training_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    training_pin_args = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    run = Run.get_context()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        if start_iter == -1:
            print("training starts from the latest iteration")
            save_list = os.listdir(system_configs.snapshot_dir)
            save_list.sort(
                key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True
            )

            print(save_list)
            if len(save_list) > 0:
                target_save = save_list[0]
                start_iter = int(re.findall(r"\d+", target_save)[0])
                learning_rate /= decay_rate ** (start_iter // stepsize)
                nnet.load_params(start_iter)
            else:
                start_iter = 0
        nnet.set_lr(learning_rate)
        print(
            "training starts from iteration {} with learning_rate {}".format(
                start_iter + 1, learning_rate
            )
        )
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()

    if not os.path.exists("data/scatterdata/scatter/outputs"):
        os.makedirs("data/scatterdata/scatter/outputs")
        print("outputs file created")
    else:
        print(os.listdir("data/scatterdata/scatter/outputs"))

    error_count = 0
    torch.cuda.empty_cache()
    for iteration in tqdm(range(start_iter + 1, max_iteration + 1)):
        try:
            # Get training data from the queue: (xs, ys). This is what in xs
            # image: (4,3,511,511) -> (batch_size, channels, height, width)
            # key_tags: (4, 1024) -> (batch_size, max_tag_len)
            # key_tags_grouped: (4, 10, 1024) -> (batch_size, max_group_len, max_tag_len_group)
            # tag_group_lens: (4)
            # tag_lens)

            training = pinned_training_queue.get(block=True)
        except:
            print("Error when extracting data")
            error_count += 1
            if error_count > 10:
                print("failed")
                time.sleep(1)
                break
            continue
        training_loss = nnet.train(**training)

        if display and iteration % display == 0:
            print(
                "training loss at iteration {}: {}".format(
                    iteration, training_loss.item()
                )
            )
            run.log("train_loss", training_loss.item())

            with open(
                "data/scatterdata/scatter/outputs/train_loss.txt", "a"
            ) as log_file:
                log_file.write(
                    f"Iteration {iteration}: Log Loss = {training_loss.item()}\n"
                )

        #
        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            nnet.eval_mode()
            validation, val_ind = sample_data(validation_db, val_ind, data_aug=False)
            validation_loss = nnet.validate(**validation)
            print(
                "validation loss at iteration {}: {}".format(
                    iteration, validation_loss.item()
                )
            )
            run.log("val_loss", validation_loss.item())
            with open("data/scatterdata/scatter/outputs/val_loss.txt", "a") as log_file:
                log_file.write(
                    f"Iteration {iteration}: Log Loss = {validation_loss.item()}\n"
                )
            nnet.train_mode()

        if iteration % snapshot == 0:
            nnet.save_params(iteration)

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)
    #
    # # sending signal to kill the thread
    training_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()


if __name__ == "__main__":
    args = parse_args()
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.data_dir + "/cache"

    file_list_data = os.listdir(args.data_dir)
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split

    print("Loading all datasets...")
    dataset = system_configs.dataset
    threads = args.threads
    print("using {} threads".format(threads))

    # Load the dataset using COCO format from db/datasets.py
    training_dbs = [
        datasets[dataset](configs["db"], train_split) for _ in range(threads)
    ]
    validation_db = datasets[dataset](configs["db"], val_split)
    # pprint.pprint(training_dbs)
    # pprint.pprint(training_dbs[0])

    print("System config...")
    ic(system_configs.full)

    print("DB config...")
    ic(f"Split: {training_dbs[0].split}")

    print(f"len of db {len(training_dbs[0].db_inds)}")
    train(training_dbs, validation_db, args.start_iter)
