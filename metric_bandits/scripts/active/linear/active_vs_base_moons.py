"""
Script for ealuation the accuracy of our models with respect to
a Linear classifier and L2 loss.
"""

from collections import defaultdict
from functools import partial
import os

import numpy as np
import torch

from metric_bandits.algos.linucb import LinUCB
from metric_bandits.envs.moons_env import MoonsSimEnv
from metric_bandits.utils.plots import plot_ci
from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.utils.read_write import save_object

# Constants for the environment
T = 4000
batch_size = 5
persistence = 1
eval_freq = 200
to_eval = ["square_loss"]
context="quadratic"

# Constants for UCB
active = True


# constants for plotting
l2_name = "LinUCB-active-vs-base-moons-square-loss"

folder = os.path.join(FIGURES_PATH, "active/linear/")
x_label = "Number of queries"

l2_y_label = "L2 distance"
# Constants for the experiment
num_trials = 10


# set up the enviromenent and model
eval_metrics = {}
for metric in to_eval:
    eval_metrics[(metric)] = defaultdict(list)


display_name = {0: "non-active", 0.1: "active"}


for explore_param in [0, 0.1]:
    for i in range(num_trials):

        print(
            f"Evaluating with explore_param={explore_param}, i={i}"
        )
        
        algo = LinUCB(
            explore_param=explore_param,
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
            context=context
        )
        env.reset()
        env.train()

        eval_metrics[("square_loss")][
            display_name[explore_param]
        ].append(env.eval_metrics["square_loss"])

current_l2_name = l2_name
l2_tuple = ("square_loss")

x_axis = (
    np.arange(len(eval_metrics[l2_tuple][display_name[explore_param]][0]))
    * eval_freq
    + eval_freq
)
l2_title = f"Square Loss"

plot_ci(
    eval_metrics[l2_tuple],
    folder,
    x_label,
    l2_y_label,
    l2_title,
    current_l2_name,
    ci=0.95,
    x_axis=x_axis,
)

save_object(eval_metrics, "big_run")
