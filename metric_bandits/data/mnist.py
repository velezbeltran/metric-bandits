"""
A simple class to keep thing uniformly formatted.
"""

import os
import pickle as pkl

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

from metric_bandits.constants.models import PCA_DIMS
from metric_bandits.constants.paths import MNIST_PATH, MNIST_PCA_MODEL_PATH

MNIST = datasets.MNIST(
    MNIST_PATH, train=True, download=True, transform=transforms.ToTensor()
)


def make_pca_mnist(mnist, n_components=PCA_DIMS):
    """
    PCA-reduces the dataset to 2 dimensions.
    """
    data, labels = [], []
    for img, label in mnist:
        data.append(img.flatten().numpy())
        labels.append(label)

    data = np.array(data)
    # center the data and scale it
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    # import model if exists otherwise make one and save
    pth = os.path.join(MNIST_PCA_MODEL_PATH, str(n_components) + ".pth")
    if os.path.exists(pth):
        pca = pkl.load(open(pth, "rb"))
    else:
        os.makedirs(MNIST_PCA_MODEL_PATH, exist_ok=True)
        pca = PCA(n_components=n_components)
        pca.fit(data)
        pkl.dump(pca, open(pth, "wb"))
    data = pca.transform(data)
    # normalize the data
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    data = [(torch.tensor(img), label) for img, label in zip(data, labels)]
    return data
