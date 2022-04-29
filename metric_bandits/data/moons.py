import numpy as np
import torch
from sklearn.datasets import make_moons


def get_moons(n_samples=60000, noise=0.2, random_state=None, balance=0.5):
    n_samples = 60000
    dummy_nsamples = int(balance * n_samples * 2)
    noise = 0.2

    X, y = make_moons(
        n_samples=dummy_nsamples, noise=noise, random_state=42, shuffle=False
    )
    X, y = X[:n_samples], y[:n_samples]

    # shuffle
    idx = np.rand.permutation(len(X))
    X, y = X[idx], y[idx]
    return torch.from_numpy(X), torch.from_numpy(y)


MOONS = get_moons()
UNBALANCED_MOONS = get_moons(n_samples=10000, balance=0.5)
