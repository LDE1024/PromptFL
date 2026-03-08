import os
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            total_train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            total_train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(total_train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        if num_shots >= 1:
            seed = cfg.SEED
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")

        # 兜底逻辑：如果未设置少样本，默认使用全量数据
        if num_shots < 1:
            train = total_train

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # 联邦数据划分逻辑
        if cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(total_train, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            print("federated all dataset")
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(total_train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            print("fewshot federated dataset")
        else:
            federated_train_x = None

        # 终极保护：防止底层 Dassl 不认识 federated_train_x 参数导致崩溃
        try:
            super().__init__(train_x=train, federated_train_x=federated_train_x, val=val, test=test)
        except TypeError:
            self.federated_train_x = federated_train_x
            super().__init__(train_x=train, val=val, test=test)

    # ---------------- 补充原作者漏写的联邦切分逻辑 ----------------
    def generate_federated_dataset(self, dataset, num_shots=-1, num_users=5, is_iid=True, repeat_rate=1.0):
        dict_users = {}
        dataset_list = list(dataset)
        random.shuffle(dataset_list)
        chunk_size = len(dataset_list) // num_users
        for i in range(num_users):
            dict_users[i] = dataset_list[i*chunk_size : (i+1)*chunk_size]
        return dict_users

    def generate_federated_fewshot_dataset(self, dataset, num_shots=-1, num_users=5, is_iid=True, repeat_rate=1.0):
        dict_users = {}
        tracker = defaultdict(list)
        for item in dataset:
            tracker[item.label].append(item)
            
        for i in range(num_users):
            user_data = []
            for label, items in tracker.items():
                if len(items) >= num_shots and num_shots > 0:
                    user_data.extend(random.sample(items, num_shots))
                else:
                    user_data.extend(items)
            dict_users[i] = user_data
        return dict_users