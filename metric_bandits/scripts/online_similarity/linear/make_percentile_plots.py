"""
Code to look at values of regime change for linucb
and display a graph for report
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric_bandits.constants.paths import FIGURES_PATH
from metric_bandits.utils.plots import colors, make_plots_pretty
from metric_bandits.utils.read_write import load_object

folder = "online_similarity/linear/"
names = ["epsilon_vs_linucb_blobs_regime_change", "epsilon_vs_linucb_blobs"]
for name in names:
    make_plots_pretty()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    obj = load_object(name)

    table = defaultdict(list)
    for i, key in enumerate(obj.keys()):
        mean = np.mean(obj[key], axis=0)[-1]
        std = np.std(obj[key], axis=0)[-1]
        ci_low = np.percentile(obj[key], 2.5, axis=0)[-1]
        ci_high = np.percentile(obj[key], 97.5, axis=0)[-1]

        table["name"].append(key)
        table["mean"].append(mean)
        table["std"].append(std)
        table["ci_low"].append(ci_low)
        table["ci_high"].append(ci_high)

        ax.scatter(
            [i],
            [np.mean(obj[key], axis=0)[-1]],
            s=100,
            c=colors[i],
            marker="o",
            label=key,
        )
        ax.errorbar(
            [i],
            [np.mean(obj[key], axis=0)[-1]],
            yerr=[[ci_low], [ci_high]],
            fmt="o",
            c=colors[i],
        )

    # adjust plot settings
    ax.set_title("Final Regret")
    ax.set_ylabel("Final Regret")
    # set the ticks to match names
    ax.set_xticks([], [])
    ax.legend(loc="upper left", prop={"size": 16})
    plt.savefig(os.path.join(FIGURES_PATH, folder, f"{name}.png"), dpi=210)

    df = pd.DataFrame(table)
    print(df)
