"""
Compares and epsilon greedy vs neural network
"""

import os
from collections import defaultdict

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
normalize = False


# Constants for the environment
T = 3000
batch_size = 7
persistence = 1
eval_freq = 200000
to_eval = []
possible_actions = [1]

# Constants for the algorithms
step_size = 0.01
num_steps = 2
train_freq = 50
verbose = False

# Constants for UCB
active = False
reset_freq = 2

# constants for plotting
folder = os.path.join(FIGURES_PATH, "online_similarity/neural/")
title = "Regret of epsilon-greedy vs neural-UCB"
name = "online_epsilon_vs_neural"
x_label = "Number of steps"
y_label = "Regret"
ci = 0.95

# Constants for experiment
num_trials = 20
greedy_explore_params = [0, 0.01, 0.1, 0.2, 0.5, 0.6, 0.1]
ucb_explore_params = [0, 0.01, 0.1, 0.5, 1, 5, 10]

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
        label = "Greedy epsilon={}".format(g_param)
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
            ucb_param,
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
