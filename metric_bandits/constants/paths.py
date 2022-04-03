"""
Contains paths to various files and directories all given with respect
to the root directory of the project.
"""
import os
from pathlib import Path

# Base paths for the project.
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")

# CIFAR_10 paths
CIFAR_10_PATH = os.path.join(DATA_PATH, "cifar-10-batches-py")
CIFAR_10_TRIPLETS_PATH = os.path.join(DATA_PATH, "cifar_triplets")

# MNIST paths
MNIST_PATH = os.path.join(DATA_PATH, "mnist")

# Model paths
MODELS = os.path.join(ROOT_PATH, "models")
MNIST_PCA_MODEL_PATH = os.path.join(MODELS, "mnist_pca_model.pt")
