"""
Script for ealuation the accuracy of our models with respect to
a KNN classifier.
"""

from collections import defaultdict

import numpy as np

from metric_bandits.algos.tlinucb import TLinUCB
from metric_bandits.envs.blobs_env import BlobsSimEnv
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import save_object

# Constants for tlinucb
input_dim = 2
reg = 1.0


# Constants for the environment
T = 1000
batch_size = 5
persistence = 2
eval_freq = 2
to_eval = ["l2_loss_linear"]
possible_actions = [1]

# Constants for UCB
active = True
num_samples = 100


# constants for plotting
l2_name = "active-vs-base-blobs-l2-loss"

folder = "active/linear/"
x_label = "Number of queries"

l2_y_label = "L2 distance"


# set up the enviromenent and model
eval_metrics = {}
for metric in to_eval:
    eval_metrics[(metric)] = defaultdict(list)


display_name = {0: "non-active", 0.1: "active"}

for i in range(num_samples):
    for explore_param in [0, 0.1]:
        print(f"Evaluating with explore_param={explore_param}, i={i}")

        algo = TLinUCB(
            input_dim=input_dim,
            explore_param=explore_param,
            active=active,
            verbose=False,
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
        eval_metrics[("l2_loss_linear")][display_name[explore_param]].append(
            env.eval_metrics["l2_loss_linear"]
        )

    current_l2_name = l2_name
    l2_tuple = "l2_loss_linear"

    x_axis = (
        np.arange(len(eval_metrics[l2_tuple][display_name[explore_param]][0]))
        * eval_freq
        + eval_freq
    )
    l2_title = "L2 distance on Blobs dataset"

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
    save_object(eval_metrics, l2_name)
