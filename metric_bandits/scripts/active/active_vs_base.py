"""
Script for ealuation the accuracy of our models with respect to
a KNN classifier.
"""

import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.envs.moons_env import MoonsSimEnv
from metric_bandits.nets.siamese import SiameseNet

# Constants for the neural network
input_dim = 2
depth = 3
hidden_dim = 25
out_dim = 2
dropout = 0.3
normalize = False


# Constants for the environment
T = 1000
batch_size = 5
persistence = 2
eval_freq = 10
to_eval = ["linear", "embedding"]

# Constants for UCB
step_size = 0.001
num_steps = 1
train_freq = 10
active = True


# set up the enviromenent and model
eval_metrics = defaultdict(list)
for explore_param in [0, 0.1]:
    for i in range(4):
        print(f"Evaluating with explore_param={explore_param}, i={i}")
        model = SiameseNet(
            input_dim, hidden_dim, depth, out_dim, dropout, normalize=normalize
        )
        algo = NeuralUCB(
            model,
            step_size,
            num_steps,
            train_freq,
            explore_param,
            active=active,
            verbose=False,
        )
        env = MoonsSimEnv(
            algo, T, batch_size, persistence, eval_freq=eval_freq, to_eval=to_eval
        )
        env.reset()
        env.train()
        eval_metrics[explore_param].append(env.eval_metrics)


# save the results
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 10})
title = "Active vs non-active learning on Moons"
name = "active_vs_base_moons_linear_clf"
pth = os.path.join(FIGURES_PATH, name + ".png")

fig, ax = plt.subplots(figsize=(10, 5))
colors = {0: "red", 0.1: "blue"}
label = {0: "Non-active", 0.1: "Active"}
for explore_param in eval_metrics.keys():
    lines = []
    for run in eval_metrics[explore_param]:
        lines.append(np.array(run["linear_acc"]))

    line = np.mean(np.array(lines), axis=0)
    x = np.arange(len(run["linear_acc"]))
    ax.plot(x, line, c=colors[explore_param], label=label[explore_param])

ax.set_title(title)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Eval Period")
ax.set_ylim(0, 1)
ax.set_xlim(0, len(x) - 1)
ax.legend()
plt.savefig(pth, dpi=300)
