"""
Script for ealuation the accuracy of our models with respect to
a KNN classifier.
"""

from collections import defaultdict
from functools import partial

import numpy as np
import torch

from metric_bandits.algos.neural_ucb import NeuralUCB
from metric_bandits.envs.moons_env import MoonsSimEnv
from metric_bandits.nets.siamese import SiameseNet
from metric_bandits.utils.plots import plot_ci
from metric_bandits.utils.read_write import save_object

# Constants for the neural network
input_dim = 2
depth = 3
hidden_dim = 25
out_dim = 3
dropout = 0.3
normalize = True

# Constants for the environment
T = 4000
batch_size = 5
persistence = 2
eval_freq = 20
to_eval = ["l2_loss_embed", "linear"]

# Constants for UCB
step_size = 0.001
num_steps = 3
train_freq = 50
active = True
reset_freq = 2
num_samples = 2
optimizers = {
    "SGD": partial(torch.optim.SGD, momentum=step_size),
    "Adam": torch.optim.Adam,
}


# constants for plotting
l2_name = "active-vs-base-moons-l2-loss"
linear_name = "active-vs-base-moons-linear-loss"

folder = "active/neural/"
x_label = "Number of queries"

l2_y_label = "L2 distance"
linear_y_label = "Accuracy"


# set up the enviromenent and model
eval_metrics = {}
for optim_name, optimizer in optimizers.items():
    for normalize in [True, False]:
        for metric in to_eval:
            eval_metrics[(optim_name, normalize, metric)] = defaultdict(list)


display_name = {0: "non-active", 0.1: "active"}
norm_str = {True: "normalized", False: "non-normalized"}

for i in range(num_samples):
    for optim_name, optimizer in optimizers.items():
        for normalize in [True, False]:
            for explore_param in [0, 0.1]:
                print(
                    f"Evaluating with explore_param={explore_param}, i={i}, optimizer={optim_name}, normalize={normalize}"
                )
                model = SiameseNet(
                    input_dim, hidden_dim, depth, out_dim, dropout, normalize=normalize
                )
                algo = NeuralUCB(
                    model,
                    step_size,
                    num_steps,
                    train_freq,
                    explore_param,
                    optimizer=optimizer,
                    active=active,
                    verbose=False,
                    reset_freq=reset_freq,
                )
                env = MoonsSimEnv(
                    algo,
                    T,
                    batch_size,
                    persistence,
                    eval_freq=eval_freq,
                    to_eval=to_eval,
                )
                env.reset()
                env.train()

                eval_metrics[(optim_name, normalize, "l2_loss_embed")][
                    display_name[explore_param]
                ].append(env.eval_metrics["l2_loss_embed"])
                eval_metrics[(optim_name, normalize, "linear")][
                    display_name[explore_param]
                ].append(env.eval_metrics["linear"])

            current_l2_name = l2_name + "-" + optim_name + "-" + norm_str[normalize]
            current_linear_name = (
                linear_name + "-" + optim_name + "-" + norm_str[normalize]
            )

            l2_tuple = (optim_name, normalize, "l2_loss_embed")
            linear_tuple = (optim_name, normalize, "linear")

            x_axis = (
                np.arange(len(eval_metrics[l2_tuple][display_name[explore_param]][0]))
                * eval_freq
                + eval_freq
            )
            l2_title = f"L2 distance with Optimizer {optim_name} and {norm_str[normalize]} vectors"
            linear_title = f"Accuracy of Linear Classifier with Optimizer {optim_name} and {norm_str[normalize]} vectors"

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
            plot_ci(
                eval_metrics[linear_tuple],
                folder,
                x_label,
                linear_y_label,
                linear_title,
                current_linear_name,
                ci=0.95,
                x_axis=x_axis,
            )
            save_object(eval_metrics, "big_run")
