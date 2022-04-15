"""
Script for evaluating neural UCB on the crescent moons dataset
generally.
"""

import os
import pickle as pkl

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import OBJECT_DUMP_PATH
from metric_bandits.envs.moons_env import MoonsSimEnv
from metric_bandits.nets.siamese import SiameseNet

# Constants for the neural network
input_dim = 2
depth = 3
hidden_dim = 25
out_dim = 2
dropout = 0.3
normalize = True


# Constants for the environment
T = 10000
batch_size = 4
persistence = 2
eval_freq = 500
possible_actions = [-1, 1]

# Constants for UCB
reg = 0.01
step_size = 0.01
num_steps = 2
train_freq = 100
explore_param = 0.1


# set up the enviromenent and model
eval_metrics = []
for i in range(2):
    model = SiameseNet(input_dim, hidden_dim, depth, out_dim, dropout, normalize)
    algo = NeuralUCB(model, reg, step_size, num_steps, train_freq, explore_param)
    env = MoonsSimEnv(
        algo,
        T,
        batch_size,
        persistence,
        eval_freq=eval_freq,
        possible_actions=possible_actions,
    )
    env.reset()
    env.train()
    eval_metrics.append(env.eval_metrics)

# save the results
title = "Crecent moons Neural-UCB Cosine Similarity"
name = "crescent_moons_neural_ucb_cosine_similarity"

pth = os.path.join(OBJECT_DUMP_PATH, name + ".pkl")
with open(pth, "wb") as f:
    pkl.dump(eval_metrics, f)

with open(pth, "rb") as f:
    eval_metrics = pkl.load(f)

# plot_regret(cum_regrets, title, name=title)
