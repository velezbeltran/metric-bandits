"""
This code contains various functions related to plotting.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from metric_bandits.constants.paths import FIGURES_PATH

# make matplotlib pretty


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


def make_plots_pretty():
    """
    Sets the standard plotting parameters
    """
    plt.style.use("ggplot")
    font = {"size": 13}
    plt.rc("font", **font)


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


def plot_ci(
    runs,
    folder,
    x_label,
    y_label,
    title="",
    name="",
    ci=0.95,
    figsize=(10, 10),
    x_axis=None,
):
    """
    Makes a plot of the lines with confidence intervals.

    Assumes that runs is a dictionary with they keys being the names of the runs
    and the values being a list of lists of the values with the same length. You can pass an $x_axis$
    value to change the values displayed on the x axis.
    """
    make_plots_pretty()
    fig, ax = plt.subplots(figsize=figsize)

    for rname, run in runs.items():
        run = np.array(run)
        run = run + np.random.normal(0, 0.01, run.shape)

        # get the confidence interval
        mean, std = np.mean(run, axis=0), np.std(run, axis=0)
        conf_int = norm.interval(ci, loc=mean, scale=std)

        # plot the confidence interval
        if x_axis is None:
            x_axis = np.arange(len(mean))
        ax.plot(x_axis, mean, label=rname)
        ax.fill_between(x_axis, conf_int[0], conf_int[1], alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if not os.path.exists(os.path.join(FIGURES_PATH, folder)):
        os.makedirs(os.path.join(FIGURES_PATH, folder))
    fig.savefig(os.path.join(FIGURES_PATH, folder, f"{name}-ci.png"), dpi=300)


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
