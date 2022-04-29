import numpy as np
import torch

N_POINTS = 100000


def get_blobs(n_points=N_POINTS, p=0.5):
    transform = np.array([[1, 0], [0, 3]])
    X = np.random.randn(n_points, 2) * 0.1
    X = np.dot(X, transform)
    Y = np.random.binomial(1, p, n_points)

    X_left = X[Y == 0]
    X_left = X_left + 1
    X_right = X[Y == 1]
    X_right[:, 0] = X_right[:, 0] + 1

    Y_left = Y[Y == 0]
    Y_right = Y[Y == 1]

    X = np.vstack((X_left, X_right))
    Y = np.hstack((Y_left, Y_right))

    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    return (torch.tensor(X), torch.tensor(Y))


BLOBS_BALANCED = get_blobs()
BLOBS_UNBALANCED = get_blobs(p=0.1)
