"""
Compares and epsilon greedy vs neural network. Like epsilon vs neural but with a slightly smaller set of parameters to look at.
"""

import os
from collections import defaultdict

import torch

from metric_bandits.algos.neural import Neural
from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.envs.mnist_env import MNISTSimEnv
from metric_bandits.nets.siamese import SiameseNet
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import load_object, save_object

load_from_file = False

# Constants for the neural network
pca_dims = 12
input_dim = pca_dims
depth = 3
hidden_dim = 25
out_dim = 5
dropout = 0.3

# Constants for the environment
T = 12000
batch_size = 12
persistence = 2
eval_freq = 200000
to_eval = []
possible_actions = [1]

# Constants for the algorithms
step_size = 0.001
num_steps = 3
train_freq = 100
verbose = False

# Constants for UCB
active = False
reset_freq = 1000
optimizer = torch.optim.Adam

# constants for plotting
folder = os.path.join(FIGURES_PATH, "online_similarity/neural/")
title = "Regret of epsilon-greedy vs neural-UCB on MNIST"
name = "online-epsilon-vs-neural-mnist"
x_label = "Number of steps"
y_label = "Regret"
ci = 0.95

# Constants for experiment
num_trials = 10
greedy_explore_params = [0, 0.01, 0.1]
ucb_explore_params = [0.01, 0.1, 0.5, 2]

if load_from_file:
    eval_metrics = load_object(name)
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")
    plot_ci(
        eval_metrics,
        folder=folder,
        title=title,
        name=name,
        x_label=x_label,
        y_label=y_label,
        ci=ci,
    )
    exit()

normalize_values = [True, False]
norm_str = {True: "normalized", False: "non-normalized"}

eval_metrics = {}
for normalize in normalize_values:
    eval_metrics[normalize] = defaultdict(list)


# first set up for epsilon greedy algorithm
for i in range(num_trials):
    for g_param, ucb_param in zip(greedy_explore_params, ucb_explore_params):
        for normalize in normalize_values:
            print(f"Evaluating greedy with explore_param={g_param}, i={i}")
            model = SiameseNet(
                input_dim, hidden_dim, depth, out_dim, dropout, normalize=normalize
            )
            algo = Neural(
                model,
                step_size,
                num_steps,
                train_freq,
                g_param,
                verbose=verbose,
            )
            env = MNISTSimEnv(
                algo,
                T,
                batch_size,
                persistence,
                eval_freq=eval_freq,
                to_eval=to_eval,
                possible_actions=possible_actions,
            )
            env.reset()
            env.train()
            label = "Greedy explore_param={}".format(g_param)
            eval_metrics[normalize][label].append(env.cum_regrets)

            print(f"Evaluating ucb with explore_param={ucb_param}, i={i}")
            model = SiameseNet(
                input_dim, hidden_dim, depth, out_dim, dropout, normalize=normalize
            )
            algo = NeuralUCB(
                model,
                step_size,
                num_steps,
                train_freq,
                explore_param=ucb_param,
                optimizer=optimizer,
                active=active,
                verbose=False,
            )
            env = MNISTSimEnv(
                algo,
                T,
                batch_size,
                persistence,
                eval_freq=eval_freq,
                to_eval=to_eval,
                possible_actions=possible_actions,
            )
            env.reset()
            env.train()
            label = "UCB explore_param={}".format(ucb_param)
            eval_metrics[normalize][label].append(env.cum_regrets)

            # save object to file
            current_name = f"{name}--{norm_str[normalize]}"
            current_title = f"{title} {norm_str[normalize]}"
            save_object(eval_metrics, current_name)
            plot_ci(
                eval_metrics[normalize],
                folder=folder,
                title=current_title,
                name=current_name,
                x_label=x_label,
                y_label=y_label,
                ci=ci,
            )
