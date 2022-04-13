"""
This code compares the performance of the algorithm with the baseline.
"""
import os
import pickle as pkl
from collections import defaultdict

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import OBJECT_DUMP_PATH
from metric_bandits.envs.mnist_env import MNISTSimEnv
from metric_bandits.nets.siamese import SiameseNet

# Constants for the neural network
pca_dims = 30
input_dim = pca_dims
depth = 3
hidden_dim = 20
out_dim = 3
dropout = 0.3

# Constants for the environment
T = 3000
batch_size = 3
persistence = 1

# Constnats for UCB
reg = 0.1
step_size = 0.01
num_steps = 20
train_freq = 50
explore_params = [0, 0.01, 0.1, 0.5, 1, 2, 5]

# set up the enviromenent and model
cum_regrets = defaultdict(list)
for explore_param in explore_params:
    for i in range(3):
        print(f"Running experiment with explore_param={explore_param} and i={i}")
        model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout)
        algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
        env = MNISTSimEnv(algo, T, batch_size, persistence, pca_dims=pca_dims)
        env.reset()
        env.train()
        env.cum_regrets = env.cum_regrets[:T]

pth = os.path.join(OBJECT_DUMP_PATH, "neural_ucb_mnist_similarity_explore_param.pkl")
with open(pth, "wb") as f:
    pkl.dump(cum_regrets, f)

with open(pth, "rb") as f:
    cum_regrets = pkl.load(f)
