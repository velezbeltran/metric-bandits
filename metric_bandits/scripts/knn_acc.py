"""
Script for ealuation the accuracy of our models with respect to
a KNN classifier.
"""

import os
import pickle as pkl

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import OBJECT_DUMP_PATH
from metric_bandits.envs.mnist_env import MNISTSimEnv
from metric_bandits.nets.siamese import SiameseNet

# Constants for the neural network
pca_dims = 25
input_dim = pca_dims
depth = 3
hidden_dim = 25
out_dim = 6
dropout = 0.3
normalize = True


# Constants for the environment
T = 10000
batch_size = 4
persistence = 2
eval_freq = 2000

# Constats for UCB
reg = 0.01
step_size = 0.01
num_steps = 2
train_freq = 150
explore_param = 0.1


# set up the enviromenent and model
eval_metrics = []
for i in range(2):
    model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout, normalize)
    algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
    env = MNISTSimEnv(
        algo, T, batch_size, persistence, pca_dims=pca_dims, eval_freq=eval_freq
    )
    env.reset()
    env.train()
    eval_metrics.append(env.eval_metrics)

# save the results
title = "Siamese network Nerual-UCB Cosine Similarity eval metrics"

pth = os.path.join(OBJECT_DUMP_PATH, title + ".pkl")
with open(pth, "wb") as f:
    pkl.dump(eval_metrics, f)

with open(pth, "rb") as f:
    eval_metrics = pkl.load(f)

# plot_regret(cum_regrets, title, name=title)
