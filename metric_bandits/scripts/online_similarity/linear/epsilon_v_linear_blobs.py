"""
Compares and epsilon-greedy Linear model vs LinUCB
"""

import os
from collections import defaultdict

from metric_bandits.algos.linear import Linear
from metric_bandits.algos.linucb import LinUCB
from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.envs.blob_env import BlobSimEnv
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import load_object, save_object

load_from_file = False

# Constants for the environment
T = 3000
batch_size = 7
persistence = 1
eval_freq = 200000
to_eval = []
possible_actions = [1]
context = "quadratic"

# Constants for the algorithms
verbose = False

# Constants for LinUCB
active = False
reset_freq = 2

# constants for plotting
folder = os.path.join(FIGURES_PATH, "online_similarity/linear/")
title = "Regret of epsilon-greedy vs Qlinear-UCB, Blob"
name = "online_epsilon_vs_qlinear_blob"
x_label = "Number of steps"
y_label = "Regret"
ci = 0.95

# Constants for experiment
num_trials = 5
greedy_explore_params = [0, 0.1, 0.5]
ucb_explore_params = [0, 0.1, 1]

if load_from_file:
    eval_metrics = load_object(name)
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
for g_param, ucb_param in zip(greedy_explore_params, ucb_explore_params):
    for i in range(num_trials):
        print(f"Evaluating greedy with explore_param={g_param}, i={i}")
        algo = Linear(
            explore_param=g_param,
            verbose=verbose
        )
        env = BlobSimEnv(
            algo,
            T,
            batch_size,
            persistence,
            eval_freq=eval_freq,
            to_eval=to_eval,
            possible_actions=possible_actions,
            context=context
        )
        env.reset()
        env.train()
        label = "Greedy epsilon={}".format(g_param)
        eval_metrics[label].append(env.cum_regrets)

        print(f"Evaluating Linucb with explore_param={ucb_param}, i={i}")
        
        algo = LinUCB(
            explore_param= ucb_param,
            active=active,
            verbose=False,
        )
        env = BlobSimEnv(
            algo,
            T,
            batch_size,
            persistence,
            eval_freq=eval_freq,
            to_eval=to_eval,
            possible_actions=possible_actions,
            context=context
        )
        env.reset()
        env.train()
        label = "UCB epsilon={}".format(ucb_param)
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
