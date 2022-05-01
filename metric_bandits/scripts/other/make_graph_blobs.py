"""
makes the graph of the blobs for the report
"""
import os

import matplotlib.pyplot as plt

from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.data.blobs import BLOBS_BALANCED, BLOBS_UNBALANCED


def make_blobs_graph(balanced, n_samples=200):
    """
    makes the graph of the blobs for the report
    """

    X, Y = BLOBS_BALANCED if balanced else BLOBS_UNBALANCED
    X = X[:n_samples]
    Y = Y[:n_samples]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="coolwarm")

    # adjust limits
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    name = "blobs-balanced" if balanced else "blobs-unbalanced"
    plt.savefig(os.path.join(FIGURES_PATH, "other", name + ".png"), dpi=210)


if __name__ == "__main__":
    make_blobs_graph(balanced=True)
    make_blobs_graph(balanced=False)
