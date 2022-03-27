import os
import pickle as pkl

import numpy as np
from torch.utils.data import Dataset

from metric_bandits.constants.constants import SEED
from metric_bandits.constants.paths import CIFAR_10_PATH


class CifarTriples(Dataset):
    def __init__(self, num_samples=1000000, verbose=False):
        self.verbose = verbose
        self.num_samples = num_samples
        self.data, self.labels, self.order = self.load_data()
        self.left_idx, self.right_idx = self.make_indices(self.num_samples)

    def load_data(self):
        files = [os.path.join(CIFAR_10_PATH, f) for f in os.listdir(CIFAR_10_PATH)]
        files = [f for f in files if not (f.endswith(".meta") or f.endswith(".html"))]
        data = []
        labels = []
        for f in files:
            f_dict = self.cifar_unpickle(f)
            data.append(f_dict[b"data"])
            labels.append(f_dict[b"labels"])
            if self.verbose:
                print("Loaded {}".format(f))

        np.random.seed(SEED)
        data = np.concatenate(data)
        labels = np.concatenate(labels)

        order = np.random.permutation(len(data))
        data = data[order]
        labels = labels[order]
        return data, labels, order

    def cifar_unpickle(self, file_name):
        with open(file_name, "rb") as fo:
            dictionary = pkl.load(fo, encoding="bytes")
        return dictionary

    def make_indices(self, num_samples):
        if self.verbose:
            print("Making indices...")

        left_proposal = np.random.choice(len(self.data), num_samples, replace=True)
        right_proposal = np.random.choice(len(self.data), num_samples, replace=True)

        has_seen = {}
        left_idx, right_idx = [], []

        if self.verbose:
            print("validating indices...")

        for i in range(num_samples):
            forder = (left_proposal[i], right_proposal[i])
            sorder = (right_proposal[i], left_proposal[i])
            if forder not in has_seen:
                left_idx.append(left_proposal[i])
                right_idx.append(right_proposal[i])
                has_seen[forder] = True
                has_seen[sorder] = True

        self.num_samples = len(left_idx)
        if self.verbose:
            print("Made indices.")
        return np.array(left_idx), np.array(right_idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        left_idx = self.left_idx[idx]
        right_idx = self.right_idx[idx]
        left_img = self.data[left_idx]
        right_img = self.data[right_idx]
        left_label = self.labels[left_idx]
        right_label = self.labels[right_idx]
        return left_img, right_img, left_label, right_label
