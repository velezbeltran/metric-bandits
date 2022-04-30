"""
Script for ealuation the accuracy of our models with respect to
a KNN classifier.
"""

from collections import defaultdict

import numpy as np

from metric_bandits.algos.linear import Linear
from metric_bandits.algos.linucb import LinUCB
from metric_bandits.envs.blobs_env import BlobsSimEnv
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import save_object

# Constants for tlinucb
input_dim = 2
reg = 1.0


# Constants for the environment
T = 3000
batch_size = 5
persistence = 2
eval_freq = 20000
possible_actions = [1]
to_eval = []

# Constants for UCB
active = False
num_samples = 30


# constants for plotting
name = "epsilon_vs_linucb_blobs"
title = "Epsilon greedy vs LinUCB"

folder = "online_similarity/linear/"
x_label = "Time"
y_label = "Regret"


# set up the enviromenent and model
eval_metrics = defaultdict(list)

explore_params_ucb = np.linspace(0.1, 4, 3)
explore_params_greedy = np.linspace(0.0, 0.3, 3)


for i in range(num_samples):
    for explore_ucb, explore_greedy in zip(explore_params_ucb, explore_params_greedy):
        print(f"Evaluating linucb with explore_param={explore_ucb}, i={i}")
        algo = LinUCB(
            input_dim=input_dim,
            explore_param=explore_ucb,
            active=active,
            verbose=False,
            reg=reg,
        )
        env = BlobsSimEnv(
            algo,
            T,
            batch_size,
            persistence,
            eval_freq=eval_freq,
            to_eval=to_eval,
            possible_actions=possible_actions,
            context="quadratic",
        )
        env.reset()
        env.train()

        linucb_string = "linucb explore_param={:.2f}".format(explore_ucb)
        eval_metrics[linucb_string].append(env.cum_regrets)

        algo = Linear(
            input_dim=input_dim,
            explore_param=explore_greedy,
            verbose=False,
            reg=reg,
        )
        env = BlobsSimEnv(
            algo,
            T,
            batch_size,
            persistence,
            eval_freq=eval_freq,
            to_eval=to_eval,
            possible_actions=possible_actions,
            context="quadratic",
        )
        env.reset()
        env.train()
        linear_string = "greedy explore_param={:.2f}".format(explore_greedy)
        eval_metrics[linear_string].append(env.cum_regrets)

        plot_ci(
            eval_metrics,
            folder,
            x_label,
            y_label,
            title=title,
            name=name,
            ci=0.95,
        )
        save_object(eval_metrics, name)
