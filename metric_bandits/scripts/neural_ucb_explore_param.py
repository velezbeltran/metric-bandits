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
from metric_bandits.utils.plots import plot_regret

# Constants for the neural network
pca_dims = 25
input_dim = pca_dims
depth = 2
hidden_dim = 20
out_dim = 4
dropout = 0.1
normalize = True


# Constants for the environment
T = 3000
batch_size = 3
persistence = 2

# Constnats for UCB
reg = 0.01
step_size = 0.01
num_steps = 10
train_freq = 50
explore_params = [0.01, 0.1, 0.5]

# set up the enviromenent and model
cum_regrets = defaultdict(list)
for explore_param in explore_params:
    for i in range(2):
        continue
        print(f"Running experiment with explore_param={explore_param} and i={i}")
        model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout, normalize)
        algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
        env = MNISTSimEnv(algo, T, batch_size, persistence, pca_dims=pca_dims)
        env.reset()
        env.train()
        cum_regrets[str(explore_param)].append(env.cum_regrets)

# save the results
title = "Siamese network Nerual-UCB Cosine Similarity"

pth = os.path.join(OBJECT_DUMP_PATH, title + ".pkl")
# with open(pth, "wb") as f:
#    pkl.dump(cum_regrets, f)

# with open(pth, "rb") as f:
#    cum_regrets = pkl.load(f)

# cum_regrets_2 = {}
# for explore_param in explore_params:
#    cum_regrets_2[str(explore_param)] = cum_regrets[str(explore_param)]

# plot_regret(cum_regrets_2, title, name=title)


# Second unormalized neural network


normalize = False
cum_regrets = defaultdict(list)
for explore_param in explore_params:
    for i in range(2):
        continue
        print(f"Running experiment with explore_param={explore_param} and i={i}")
        model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout, normalize)
        algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
        env = MNISTSimEnv(algo, T, batch_size, persistence, pca_dims=pca_dims)
        env.reset()
        env.train()
        cum_regrets[str(explore_param)].append(env.cum_regrets)

# save the results
pth = os.path.join(OBJECT_DUMP_PATH, title + ".pkl")
# with open(pth, "wb") as f:
# pkl.dump(cum_regrets, f)

with open(pth, "rb") as f:
    cum_regrets = pkl.load(f)

title = "Siamese network Nerual-UCB Dot Product Similarity"
plot_regret(cum_regrets, title, name=title)
