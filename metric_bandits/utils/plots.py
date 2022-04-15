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
font = {"size": 9}
plt.rc("font", **font)

colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "w",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_regret(regrets, folder, title="", name=""):
    """
    Plots the cumulative regret and saves it in the figures directory.
    Assumes that regrets is a dictionary with they keys being the names
    and the values being a list of lists of cummunative regrets.
    """
    fig, ax = plt.subplots()
    c = 0
    for rname, regret in regrets.items():
        regret = torch.tensor(regret).numpy()
        for i in range(len(regret)):
            ax.plot(np.arange(len(regret[i])), regret[i], color=colors[c])
        ax.plot(np.arange(len(regret[i])), regret[i], label=rname, color=colors[c])
        c += 1

    ax.set_title("Regret " + title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_PATH, folder, f"{name}_regret.png"), dpi=300)


def plot_embeddings(embeddings, labels, folder, title="", name=""):
    """
    Plots the embeddings and saves it in the figures directory.
    Embeddings have shape [num_embeddings, embedding_dim]
    and the points in scatter plot are colored by labels
    """
    fig, ax = plt.subplots()

    for i in range(embeddings.shape[0]):
        ax.scatter(
            embeddings[i, 0], embeddings[i, 1], color=colors[labels[i]], alpha=0.5
        )

    ax.set_title("Embeddings " + title)
    fig.savefig(os.path.join(FIGURES_PATH, f"{name}_embeddings.png"), dpi=300)
