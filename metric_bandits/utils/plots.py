"""
This code contains various functions related to plotting.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from metric_bandits.constants.paths import FIGURES_PATH

# make matplotlib pretty
plt.style.use("ggplot")

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]


def plot_regret(regrets, name=""):
    """
    Plots the cumulative regret and saves it in the figures directory.
    Assumes that regrets is a dictionary with they keys being the names
    and the values being a list of lists of cummunative regrets.
    """
    fig, ax = plt.subplots()
    c = 0
    for name, regret in regrets.items():
        regret = torch.tensor(regret).to_numpy()
        for i in range(len(regret)):
            ax.plot(np.arange(len(regret[i])), regret[i], label=name, color=colors[c])
        c += 1

    ax.set_title("Regret")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_PATH, f"{name}_regret.png"), dpi=300)
