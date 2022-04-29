import numpy as np
import torch

N_POINTS = 100000

transform = np.array([[1, 0], [0, 3]])
X = np.random.randn(N_POINTS, 2) * 0.1
X = np.dot(X, transform)
Y = np.random.binomial(1, 0.5, N_POINTS)

X_left = X[Y == 0]
X_left[:, 0] = X_left[:, 0] - 1
X_right = X[Y == 1]
X_right[:, 0] = X_right[:, 0] + 1

Y_left = Y[Y == 0]
Y_right = Y[Y == 1]

X = np.vstack((X_left, X_right))
Y = np.hstack((Y_left, Y_right))

BLOBS = (torch.tensor(X), torch.tensor(Y))
