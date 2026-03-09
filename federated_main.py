import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import time

import os
import copy
import numpy as np
from tqdm import tqdm

from utils import get_dataset, average_weights, exp_details, count_parameters
from draw import visualize

# custom datasets
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.food101
import datasets.caltech101

# custom trainers
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.promptfl


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    # PromptFL config
    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = 16
    cfg.TRAINER.PROMPTFL.CSC = False
    cfg.TRAINER.PROMPTFL.CTX_INIT = ""
    cfg.TRAINER.PROMPTFL.PREC = "fp16"
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"

    # dataset / FL config
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.USERS = 2
    cfg.DATASET.IID = False
    cfg.DATASET.USEALL = False
    cfg.DATASET.REPEATRATE = 0.0
    cfg.OPTIM.ROUND = 10
    cfg.OPTIM.MAX_EPOCH = 5

    cfg.MODEL.BACKBONE.PRETRAINED = True

    # attack config
    cfg.ATTACK = CN()
    cfg.ATTACK.ENABLE = False
    cfg.ATTACK.ATTACKER_ID = 0
    cfg.ATTACK.TARGET_LABEL = 0
    cfg.ATTACK.POISON_RATIO = 0.4
    cfg.ATTACK.LAMBDA = 1.0
    cfg.ATTACK.EPS = 8.0 / 255.0


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. dataset config
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. method config
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. command line args
    reset_cfg(cfg, args)

    # 4. extra opts
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    global_trainer = build_trainer(cfg)
    print("type", type(global_trainer))
    global_trainer.fed_before_train(is_global=True)

    # global weights
    global_weights = global_trainer.model.state_dict()

    # local trainer
    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()

    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND

    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    global_asr_list = []

    start = time.time()

    for epoch in range(start_epoch, max_epoch):
        local_weights = []

        idxs_users = list(range(0, cfg.DATASET.USERS))
        print("idxs_users", idxs_users)
        print("------------local train start epoch:", epoch, "-------------")

        for idx in idxs_users:
            local_trainer.model.load_state_dict(global_weights)

            # default: no attack
            local_trainer.attack_enable = cfg.ATTACK.ENABLE
            local_trainer.is_malicious = False

            # set attack params for malicious client
            if cfg.ATTACK.ENABLE and idx == cfg.ATTACK.ATTACKER_ID:
                print(f"[!!!] Client {idx} is executing route-A backdoor attack")
                local_trainer.is_malicious = True
                local_trainer.poison_ratio = cfg.ATTACK.POISON_RATIO
                local_trainer.target_label = cfg.ATTACK.TARGET_LABEL
                local_trainer.attack_lambda = cfg.ATTACK.LAMBDA
                local_trainer.eps = cfg.ATTACK.EPS

            print(
                f"[CLIENT {idx}] attack_enable={local_trainer.attack_enable}, "
                f"is_malicious={local_trainer.is_malicious}"
            )

            local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)

            local_weight = local_trainer.model.state_dict()
            local_weights.append(copy.deepcopy(local_weight))

        print("------------local train finish epoch:", epoch, "-------------")

        # aggregate
        global_weights = average_weights(local_weights)
        global_trainer.model.load_state_dict(global_weights)

        print("------------global test start-------------")
        result = global_trainer.test(is_global=True, current_epoch=epoch)

        global_test_acc_list.append(result[0])
        global_test_error_list.append(result[1])
        global_test_f1_list.append(result[2])
        global_epoch_list.append(epoch)
        global_time_list.append(time.time() - start)

        if cfg.ATTACK.ENABLE:
            asr_score = global_trainer.test_asr()
            global_asr_list.append(asr_score)
            print(
                f"-> Global Round: {epoch} | Main Accuracy: {result[0]:.2f}% "
                f"| ASR: {asr_score:.2f}%"
            )
        else:
            print(f"-> Global Round: {epoch} | Main Accuracy: {result[0]:.2f}%")

        print("------------global test finish-------------")

    local_trainer.fed_after_train()
    global_trainer.fed_after_train()

    # visualize
    if cfg.ATTACK.ENABLE:
        visualize(
            global_test_acc_list,
            global_test_error_list,
            global_test_f1_list,
            global_epoch_list,
            global_time_list,
            cfg.OUTPUT_DIR,
            asr_list=global_asr_list,
        )
    else:
        visualize(
            global_test_acc_list,
            global_test_error_list,
            global_test_f1_list,
            global_epoch_list,
            global_time_list,
            cfg.OUTPUT_DIR,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
