"""
Compares and epsilon greedy vs neural network. Like epsilon vs neural but with a slightly smaller set of parameters to look at.
"""

import os
from collections import defaultdict

import torch

from metric_bandits.algos.neural import Neural
from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.envs.moons_env import MoonsSimEnv
from metric_bandits.nets.siamese import SiameseNet
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import load_object, save_object

load_from_file = False

# Constants for the neural network
input_dim = 2
depth = 3
hidden_dim = 25
out_dim = 3
dropout = 0.3
normalize = True

# Constants for the environment
T = 4000
batch_size = 10
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
reset_freq = 10000000
optimizer = torch.optim.Adam

# constants for plotting
folder = os.path.join(FIGURES_PATH, "online_similarity/neural/")
title = "Regret of epsilon-greedy vs neural-UCB "
name = "online-epsilon-vs-neural-reduced-normalized"
x_label = "Number of steps"
y_label = "Regret"
ci = 0.95

# Constants for experiment
num_trials = 10
greedy_explore_params = [0, 0.01, 0.1]
ucb_explore_params = [0, 0.01, 0.1, 0.5, 2]

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


eval_metrics = defaultdict(list)
# first set up for epsilon greedy algorithm
for i in range(num_trials):
    for g_param, ucb_param in zip(greedy_explore_params, ucb_explore_params):
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
        env = MoonsSimEnv(
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
        eval_metrics[label].append(env.cum_regrets)

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
        env = MoonsSimEnv(
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
        eval_metrics[label].append(env.cum_regrets)

        # save object to file
        save_object(eval_metrics, name)
        plot_ci(
            eval_metrics,
            folder=folder,
            title=title,
            name=name,
            x_label=x_label,
            y_label=y_label,
            ci=ci,
        )
